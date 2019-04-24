#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <openssl/md5.h>
#include <cuda.h>
#include "md5.cu"

__global__ void hash_it_up (char** output, int max){
  char word[] = "aaaaaa";
  int i = 0;
  for(int tempNum = 0; tempNum < max; tempNum+=5){

    //Generate the permutation for the given thread/core
    for(int i = PASSWORD_LENGTH - 1; i >= 0; i--){
      int temp = (int) (tempNum/POWER_ARR[i]);
      word[5 - i] += temp;
      tempNum = tempNum % POWER_ARR[i];
    }

    //we need to add hashes to ouput and write to file in main
  }

  
  
}

int main(int argc,char* args[]){
  
  
  return 0;
}
