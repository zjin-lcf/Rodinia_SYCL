#ifndef _COMPARISON_HELPERS_H_
#define _COMPARISON_HELPERS_H_

template <typename T>
__inline int compare_vectors(T* data1, T* data2, unsigned int size) {
	printf("Comparing vectors of length %u: \n", size);
	bool match = true;
	for(unsigned int i = 0; i < size; i++)  
		if (data1[i]!= data2[i]) {
			match = false;
			printf("Diff: reference data[%d]=%u,  device data[%d]=%u.\n",
          i,data1[i],i,data2[i]);
      break; // early stop
		}

	if (match) { printf("PASS! vectors are matching!\n"); return 0;	}
	else {printf("FAIL! vectors are NOT matching!\n");	return -1; }
}

#endif
