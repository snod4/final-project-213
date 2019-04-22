#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define NUM_THREADS = 32
#define MAX_USERNAME_LENGTH 64

#define PASSWORD_LENGTH 6
#define FIFTH_POWER (26 * 26 * 26 * 26 * 26)
#define FOURTH_POWER (26 * 26 * 26 * 26)
#define THIRD_POWER (26 * 26 * 26)
#define SECOND_POWER (26 * 26)
#define FIRST_POWER 26



int POWER_ARR[] = {1, FIRST_POWER, SECOND_POWER, THIRD_POWER, FOURTH_POWER, FIFTH_POWER};


typedef struct hashInfo{
  char * password;
  char * hash;
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

  //-----HASH CODE HERE-----//
  char * hashVar;
  for(int i = 0; i < length, i++){
    
  }
  

  
  // strncpy(output, word, PASSWORD_LENGTH+1);


  
}

int main(){

  return 0;
}
