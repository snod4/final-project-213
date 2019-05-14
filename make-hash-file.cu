#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <openssl/md5.h>
#include <cuda.h>
#include "md5.cu"

#define PASSWORD_LENGTH 6

#define FIFTH_POWER (26 * 26 * 26 * 26 * 26)
#define FOURTH_POWER (26 * 26 * 26 * 26)
#define THIRD_POWER (26 * 26 * 26)
#define SECOND_POWER (26 * 26)
#define FIRST_POWER 26 

__device__ int POWER_ARR[] = {1, FIRST_POWER, SECOND_POWER, THIRD_POWER, FOURTH_POWER, FIFTH_POWER};

typedef struct output{
  char password[PASSWORD_LENGTH+1];
  uint hash[4];
}output_t;


/*hash_it_up takes a pointer to output and feeds out 'max' passwords and
hashes. Runs on the GPU*/
__global__ void hash_it_up (output_t * output, int max){
  
  int tempNum;
  for(int count = 0; count < max; count++){
    char word[] = "aaaaaa";
    tempNum = count;

    //Generate and store a permutation
    for(int i = PASSWORD_LENGTH - 1; i >= 0; i--){
      int temp = (int) (tempNum/POWER_ARR[i]);
      word[PASSWORD_LENGTH -1 - i] += temp;
      tempNum = tempNum % POWER_ARR[i];
    }
    
    
    word[PASSWORD_LENGTH] = '\0';
    
    //Store the Password
    memcpy(output[count].password, word, sizeof(char)*PASSWORD_LENGTH+1);

    //Hash a pasword
    uint candidate_hash[4];
    md5((uint*)word, candidate_hash);

    //Store the hash
    memcpy(output[count].hash, candidate_hash, sizeof(uint)*4);
  }

  
  
}

int main(int argc,char* args[]){
  FILE * file;
  int max = atoi(args[1]);
  output_t * gpu_input;
  output_t * output = (output_t *) malloc(sizeof(output_t)*max);

  //Allocate space on GPU
  if(cudaMalloc(&gpu_input, sizeof(output_t)*max) != cudaSuccess){
    perror("Cuda Malloc Failed\n");
  }

  //Call the hash-generating function
  hash_it_up<<<1,1>>>(gpu_input, max);

  //Wait for all of the threads to finish
  if(cudaDeviceSynchronize() != cudaSuccess){
    perror("Cuda Sync Failed\n");
  }

  //Copy back the generated passwords from the GPU
  if(cudaMemcpy(output, gpu_input, sizeof(output_t)*max,
                cudaMemcpyDeviceToHost) != cudaSuccess){
    perror("Cuda Memcpy Failed Here\n");
    exit(2);
  }

  
  //Write passwords and hashes to a file
  file = fopen("outputFile.txt", "w");
  fprintf(file, "%d\n", max);
  for(int i = 0; i < max; i++){
    fprintf(file, "%s ",output[i].password);
    for(int j = 0; j < 4; j++){
     fprintf(file, "%u ",output[i].hash[j]);
    }
    fprintf(file, "\n");
  }
  fclose(file);
  
  
  return 0;
}
