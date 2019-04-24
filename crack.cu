#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <openssl/md5.h>
#include <cuda.h>
#include "md5.cu"

#define _GNU_SOURCE
#define NUM_THREADS = 32
#define MAX_USERNAME_LENGTH 64

#define PASSWORD_LENGTH 6
#define FIFTH_POWER (26 * 26 * 26 * 26 * 26)
#define FOURTH_POWER (26 * 26 * 26 * 26)
#define THIRD_POWER (26 * 26 * 26)
#define SECOND_POWER (26 * 26)
#define FIRST_POWER 26

#define MD5_DIGEST_LENGTH
#define HASH_LENGTH 32



const int POWER_ARR[] = {1, FIRST_POWER, SECOND_POWER, THIRD_POWER, FOURTH_POWER, FIFTH_POWER};


typedef struct hashInfo{
  char * password;
  char * hash;
  struct * hashInfo next;
}hashInfo_t;



__global__  void crack(hashInfo_t * hashData, int length){
  //get string permuation
  int tempNum = blockIdx.x * NUM_THREADS + threadIdx.x;
  char word[] = "aaaaaa";

  //Generate the permutation for the given thread/core
  for(int i = PASSWORD_LENGTH - 1; i >= 0; i--){
    int temp = (int) (tempNum/POWER_ARR[i]);
    word[5 - i] += temp;
    tempNum = tempNum % POWER_ARR[i];
  }
  char output[PASSWORD_LENGTH];
  
  

  //-----HASH CODE HERE-----//
  char * hashVar;

  uint* candidate_hash;
  md5((uint*)word, candidate_hash);
  
  
  for(int j = 0; j < 100; j++){
    //add one for NULL character?
    if(memcmp(hashData[j], candidate_hash, HASH_LENGTH) == 0){
      cudaMemcpy(hashData->password, word, PASSWORD_LENGTH, cudaMemcpyDeviceToDevice);
      break;
      }
  }
  

  
  // strncpy(output, word, PASSWORD_LENGTH+1);


  
}

//add hash to hash table
void addToTable(hashInfo * table, char * hash){
  hashInfo_t * temp = (hashInfo_t *) malloc(sizeof(hashInfo_t));
  strncpy(temp->hash, hash, PASSWORD_LENGTH);
  temp->next = NULL;
  temp->password = NULL;
  
  if(table[hash[0] - 48 ]== NULL){
    table[hash[0] - 48] = temp;
  }
  else{
    temp->next =  table[hash[0] - 48];
    table[hash[0] - 48] = temp;
  }
}

int main(int, argv, char* args[]){
  /*  hashInfo_t * hashTable[74];
  int count = 0;
  //get hashes in here -- add them -- count them //
  
  hashInfo_t * gpu_hashTable;
 
  //ISSUE IN COPYING A LINKED LIST TO THE GPU
 */
  int number_of_blocks = (100+NUM_THREADS)/NUM_THREADS;
  hashInfo_t arr[100];
  if(argv != 2){
    perror("crack <File Path>\n");
    exit(2);
  }
  FILE * file = fopen(args[1], "r");
  char * hash;
  char * outputHash;
  int count;
  while(fgets(&hash, HASH_LENGTH+1, file) && count < 100){
    outputHash = strsep(&hash, "\n");
    arr[count].hash =  strdup(outputHash, strlen(outputHash));
    count++;
  }

  hashInfo_t * gpu_arr;
  if(cudaMalloc(gpu_arr, arr, sizeof(hashInfo_t) * 100) != cudaSuccess){
    perror("Cuda Malloc Failed\n");
    exit(2);
  }

  if(cudaMemcpy(gpu_arr, arr, sizeof(hashInfo_t) * 100, cudaMemcpyHostToDevice) != cudaSuccess){
    perror("Cuda CPU to GPU memcpy Failed\n");
    exit(2);
  }
  
  crack<<<number_of_blocks, NUM_THREADS>>>(gpu_arr,100);

  if(cudaDeviceSynchronize() != cudaSuccess){
    perror("CUDA Thread Synchronization Error\n");
    exit(2);
  }

  if(cudaMemcpy(arr, gpu_arr, sizeof(hashInfo_t) * 100, cudaMemcpyDeviceToHost) != cudaSuccess){
    perror("Cuda GPU to CPU memcpy Failed\n");
    exit(2);
  }

  for(int i - 0; i < 100; i++){
    printf("Password: %s, Hash: %s\n", arr[i].password, arr[i].hash);
  }

  
  

  
  
  return 0;
}
