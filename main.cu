#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <openssl/md5.h>

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



const int POWER_ARR[] = {1, FIRST_POWER, SECOND_POWER, THIRD_POWER, FOURTH_POWER, FIFTH_POWER};


typedef struct hashInfo{
  char * password;
  char * hash;
  struct * hashInfo next;
}hashInfo_t;



__global__  void crack(char * startString, hashInfo_t * hashData, int length){
  //get string permuation
  int stringPerm = blockIdx.x * NUM_THREADS + threadIdx.x;
  int tempNum = stringPerm;
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
  
  while(hashData->next != NULL){
    if(memcmp(hashData->hash, hashVar, MD5_DIGEST_LENGTH) == 0){
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

int main(){
  hashInfo_t * hashTable[74];
  int count = 0;
  //get hashes in here -- add them -- count them //
  
  hashInfo_t * gpu_hashTable;
  int number_of_blocks = (count+NUM_THREADS)/NUM_THREADS;
  //ISSUE IN COPYING A LINKED LIST TO THE GPU 


  

  
  
  return 0;
}
