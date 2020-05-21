/**   Edited by: Shuai Che, David Tarjan, Sang-Ha Lee					**/
/**				 University of Virginia									**/
/**																		**/
/**   Description:	No longer supports fuzzy c-means clustering;	 	**/
/**					only regular k-means clustering.					**/
/**					No longer performs "validity" function to analyze	**/
/**					compactness and separation crietria; instead		**/
/**					calculate root mean squared error.					**/
/**                                                                     **/
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include "kmeans.h"

float	min_rmse_ref = FLT_MAX;		
extern double wtime(void);
/* reference min_rmse value */

/*---< cluster() >-----------------------------------------------------------*/
int cluster(int      npoints,				/* number of data points */
    int      nfeatures,				/* number of attributes for each point */
    float  **features,			/* array: [npoints][nfeatures] */                  
    int      min_nclusters,			/* range of min to max number of clusters */
    int		 max_nclusters,
    float    threshold,				/* loop terminating factor */
    int     *best_nclusters,		/* out: number between min and max with lowest RMSE */
    float ***cluster_centres,		/* out: [best_nclusters][nfeatures] */
    float	*min_rmse,				/* out: minimum RMSE */
    int		 isRMSE,				/* calculate RMSE */
    int		 nloops					/* number of iteration for each number of clusters */
    )
{    
  int		index =0;						/* number of iteration to reach the best RMSE */
  int		rmse;							/* RMSE for each clustering */
  float delta;

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  //queue q(dev_sel, exception_handler);
  queue q(dev_sel);


  /* current memberships of points  */
  int *membership = (int*) malloc(npoints * sizeof(int));

  /* new memberships of points computed by the device */
  int *membership_OCL = (int*) malloc(npoints * sizeof(int));

  // associated with feature[0]
  buffer<float, 1>d_feature(features[0], range<1>(npoints * nfeatures), props);

  // d_feature_swap is written by the first kernel, and read by the second kernel 
  buffer<float, 1>d_feature_swap(range<1>(npoints * nfeatures));
  
  buffer<int, 1>d_membership((range<1>(npoints)));


  // set the global work size based on local work size
  size_t global_work_size = npoints;
  size_t local_work_size= BLOCK_SIZE; // work group size is defined by RD_WG_SIZE_0 or RD_WG_SIZE_0_0 2014/06/10 17:00:51
  if(global_work_size%local_work_size !=0)
    global_work_size=(global_work_size/local_work_size+1)*local_work_size;


  /* sweep k from min to max_nclusters to find the best number of clusters */
  for(int nclusters = min_nclusters; nclusters <= max_nclusters; nclusters++)
  {
    if (nclusters > npoints) break;	/* cannot have more clusters than points */

    int c = 0;  // for each cluster size, count the actual number of loop interations

    // copy the feature to a feature swap region
    q.submit([&](handler& cgh) {
        auto acc_feature = d_feature.get_access<sycl_read>(cgh);
        auto acc_feature_swap = d_feature_swap.get_access<sycl_write>(cgh);

        cgh.parallel_for<class kernel2>(
            nd_range<1>(range<1>(global_work_size), range<1>(local_work_size)), [=] (nd_item<1> item) {
              unsigned int tid = item.get_global_id();
              if (tid < npoints) {
                for(int i = 0; i <  nfeatures; i++)
                  acc_feature_swap[i * npoints + tid] = acc_feature[tid * nfeatures + i];
                }
            });
        });

    // create clusters of size 'nclusters'
    float** clusters;
    clusters    = (float**) malloc(nclusters *             sizeof(float*));
    clusters[0] = (float*)  malloc(nclusters * nfeatures * sizeof(float));
    for (int i=1; i<nclusters; i++) clusters[i] = clusters[i-1] + nfeatures;

    /* initialize the random clusters */
    int* initial = (int *) malloc (npoints * sizeof(int));
    for (int i = 0; i < npoints; i++) initial[i] = i;
    int initial_points = npoints;

    /* iterate nloops times for each number of clusters */
    for(int lp = 0; lp < nloops; lp++)
    {
      int n = 0;

      //if (nclusters > npoints) nclusters = npoints;

      /* pick cluster centers based on the initial array 
         Maybe n = (int)rand() % initial_points; is more straightforward
         without using the initial array
       */	
      for (int i=0; i<nclusters && initial_points >= 0; i++) {

        for (int j=0; j<nfeatures; j++)
          clusters[i][j] = features[initial[n]][j];	// remapped

        /* swap the selected index with the end index. For the next iteration
           of nloops, initial[0] is differetn from initial[0] in the previous iteration */

        int temp = initial[n];
        initial[n] = initial[initial_points-1];
        initial[initial_points-1] = temp;
        initial_points--;
        n++;
      }

      /* initialize the membership to -1 for all */
      for (int i=0; i < npoints; i++) membership[i] = -1;

      /* allocate space for and initialize new_centers_len and new_centers */
      int* new_centers_len = (int*) calloc(nclusters, sizeof(int));
      float** new_centers    = (float**) malloc(nclusters *            sizeof(float*));
      new_centers[0] = (float*)  calloc(nclusters * nfeatures, sizeof(float));
      for (int i=1; i<nclusters; i++) new_centers[i] = new_centers[i-1] + nfeatures;


      //printf("global work size = %d local work size = %d\n", global_work_size, local_work_size);
      /* iterate until convergence */
      buffer<float, 1>d_clusters(range<1>(nclusters * nfeatures));

      int loop = 0;
      do {
        delta = 0.0;

        q.submit([&](handler& cgh) {
           auto acc_clusters = d_clusters.get_access<sycl_write>(cgh);
           cgh.copy(clusters[0], acc_clusters);
        });

        q.submit([&](handler& cgh) {
            auto acc_feature    = d_feature_swap.get_access<sycl_read>(cgh);
            auto acc_clusters   = d_clusters.get_access<sycl_read>(cgh);
            auto acc_membership = d_membership.get_access<sycl_write>(cgh);

            cgh.parallel_for<class kernel_s>(
                nd_range<1>(range<1>(global_work_size), range<1>(local_work_size)), 
                [=] (nd_item<1> item) 
            {
                unsigned int point_id = item.get_global_id();
                int index = 0;
                if (point_id < npoints) {
                  float min_dist=FLT_MAX;
                  for (int i=0; i < nclusters; i++) {
                    float dist = 0;
                    float ans  = 0;
                    for (int l=0; l< nfeatures; l++) {
                      ans += (acc_feature[l*npoints+point_id] - acc_clusters[i*nfeatures+l])* 
                             (acc_feature[l*npoints+point_id] - acc_clusters[i*nfeatures+l]);
                    }
                    dist = ans;
                    if (dist < min_dist) {
                      min_dist = dist;
                      index    = i;
                    }
                  }
                  acc_membership[point_id] = index;
                }

            });
        });

        q.submit([&](handler& cgh) {
           auto acc_membership = d_membership.get_access<sycl_read>(cgh);
           cgh.copy(acc_membership, membership_OCL);
        });

        q.wait();

        /* 
           1 compute the 'new' size and center of each cluster 
           2 update the membership of each point. 
         */

        for (int i = 0; i < npoints; i++)
        {
          int cluster_id = membership_OCL[i];
          new_centers_len[cluster_id]++;
          if (membership_OCL[i] != membership[i])
          {
            delta++;
            membership[i] = membership_OCL[i];
          }
          for (int j = 0; j < nfeatures; j++)
          {
            new_centers[cluster_id][j] += features[i][j];
          }
        }

        /* replace old cluster centers with new_centers */
        for (int i=0; i<nclusters; i++) {
          //printf("length of new cluster %d = %d\n", i, new_centers_len[i]);
          for (int j=0; j<nfeatures; j++) {
            if (new_centers_len[i] > 0)
              clusters[i][j] = new_centers[i][j] / new_centers_len[i];	/* take average i.e. sum/n */
            new_centers[i][j] = 0.0;	/* set back to 0 */
          }
          new_centers_len[i] = 0;			/* set back to 0 */
        }	 
        c++;
      } while ((delta > threshold) && (loop++ < 500));	/* makes sure loop terminates */

      free(new_centers[0]);
      free(new_centers);
      free(new_centers_len);

      // free the previous cluster centers before using the updated one
      if (*cluster_centres) {
        free((*cluster_centres)[0]);
        free(*cluster_centres);
      }
      *cluster_centres = clusters;

      /* find the number of clusters with the best RMSE */
      if(isRMSE)
      {
        rmse = rms_err(features,
            nfeatures,
            npoints,
            clusters,
            nclusters);

        if(rmse < min_rmse_ref){
          min_rmse_ref = rmse;			//update reference min RMSE
          *min_rmse = min_rmse_ref;		//update return min RMSE
          *best_nclusters = nclusters;	//update optimum number of clusters
          index = lp;						//update number of iteration to reach best RMSE
        }
      }
    }

    //free(clusters[0]);
    //free(clusters);
    free(initial);
  }
  free(membership_OCL);
  free(membership);

  return index;
}

