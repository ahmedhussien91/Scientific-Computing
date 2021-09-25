#include <iostream>


#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include "string.h"
#include <chrono>

using namespace std;
using namespace std::chrono;


#define DEFAULT_THRESHOLD  4000

#define DEFAULT_FILENAME "GS_1024x1024.ppm"//baboon.pgm"//




unsigned int *read_ppm( char *filename, int * xsize, int * ysize, int *maxval ){
  
  if ( !filename || filename[0] == '\0') {
    fprintf(stderr, "read_ppm but no file name\n");
    return NULL;  // fail
  }

  FILE *fp;

  fprintf(stderr, "read_ppm( %s )\n", filename);
  fp = fopen( filename, "rb");
  if (!fp) 
    {
      fprintf(stderr, "read_ppm()    ERROR  file '%s' cannot be opened for reading\n", filename);
      return NULL; // fail 

    }

  char chars[1024];
  //int num = read(fd, chars, 1000);
  int num = fread(chars, sizeof(char), 1000, fp);

  if (chars[0] != 'P' || chars[1] != '6') 
    {
      fprintf(stderr, "Texture::Texture()    ERROR  file '%s' does not start with \"P6\"  I am expecting a binary PPM file\n", filename);
      return NULL;
    }

  unsigned int width, height, maxvalue;


  char *ptr = chars+3; // P 6 newline
  if (*ptr == '#') // comment line! 
    {
      ptr = 1 + strstr(ptr, "\n");
    }

  num = sscanf(ptr, "%d\n%d\n%d",  &width, &height, &maxvalue);
  fprintf(stderr, "read %d things   width %d  height %d  maxval %d\n", num, width, height, maxvalue);  
  *xsize = width;
  *ysize = height;
  *maxval = maxvalue;
  
  unsigned int *pic = (unsigned int *)malloc( width * height * sizeof(unsigned int));
  if (!pic) {
    fprintf(stderr, "read_ppm()  unable to allocate %d x %d unsigned ints for the picture\n", width, height);
    return NULL; // fail but return
  }

  // allocate buffer to read the rest of the file into
  int bufsize =  3 * width * height * sizeof(unsigned char);
  if ((*maxval) > 255) bufsize *= 2;
  unsigned char *buf = (unsigned char *)malloc( bufsize );
  if (!buf) {
    fprintf(stderr, "read_ppm()  unable to allocate %d bytes of read buffer\n", bufsize);
    return NULL; // fail but return
  }





  // TODO really read
  char duh[80];
  char *line = chars;

  // find the start of the pixel data.   no doubt stupid
  sprintf(duh, "%d\0", *xsize);
  line = strstr(line, duh);
  //fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
  line += strlen(duh) + 1;

  sprintf(duh, "%d\0", *ysize);
  line = strstr(line, duh);
  //fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
  line += strlen(duh) + 1;

  sprintf(duh, "%d\0", *maxval);
  line = strstr(line, duh);


  fprintf(stderr, "%s found at offset %d\n", duh, line - chars);
  line += strlen(duh) + 1;

  long offset = line - chars;
  //lseek(fd, offset, SEEK_SET); // move to the correct offset
  fseek(fp, offset, SEEK_SET); // move to the correct offset
  //long numread = read(fd, buf, bufsize);
  long numread = fread(buf, sizeof(char), bufsize, fp);
  fprintf(stderr, "Texture %s   read %ld of %ld bytes\n", filename, numread, bufsize); 

  fclose(fp);


  int pixels = (*xsize) * (*ysize);
  for (int i=0; i<pixels; i++) pic[i] = (int) buf[3*i];  // red channel

 

  return pic; // success
}





void write_ppm( char *filename, int xsize, int ysize, int maxval, int *pic) 
{
  FILE *fp;
  int x,y;
  
  fp = fopen(filename, "w");
  if (!fp) 
    {
      fprintf(stderr, "FAILED TO OPEN FILE '%s' for writing\n");
      exit(-1); 
    }
  
  
  
  fprintf(fp, "P6\n"); 
  fprintf(fp,"%d %d\n%d\n", xsize, ysize, maxval);
  
  int numpix = xsize * ysize;
  for (int i=0; i<numpix; i++) {
    unsigned char uc = (unsigned char) pic[i];
    fprintf(fp, "%c%c%c", uc, uc, uc); 
  }
  fclose(fp);

}




void HostPrewitt(int xsize, int ysize,int thresh, unsigned int *input_pic, int *output)
{
    int i, j, magnitude, sum1, sum2;

    for (i = 1;  i < ysize - 1; i++) {
      for (j = 1; j < xsize -1; j++) {

        int offset = i*xsize + j;

        sum1 =  input_pic[ xsize * (i-1) + j+1 ] -     input_pic[ xsize*(i-1) + j-1 ] 
          +     input_pic[ xsize * (i)   + j+1 ] -     input_pic[ xsize*(i)   + j-1 ]
          +     input_pic[ xsize * (i+1) + j+1 ] -     input_pic[ xsize*(i+1) + j-1 ];

        sum2 = input_pic[ xsize * (i-1) + j-1 ] +     input_pic[ xsize * (i-1) + j ]  + input_pic[ xsize * (i-1) + j+1 ]
              - input_pic[xsize * (i+1) + j-1 ] -     input_pic[ xsize * (i+1) + j ] - input_pic[ xsize * (i+1) + j+1 ];

        magnitude =  sum1*sum1 + sum2*sum2;

        if (magnitude > thresh)
          output[offset] = 255;
        else 
          output[offset] = 0;
      }
    }
}


__global__ void gpuPrewittWithoutSm(int xsize, int ysize,int thresh, unsigned int *input_pic, int *result)
{
    int by = blockIdx.x;  int bx = blockIdx.y;
    int ty = threadIdx.x; int tx = threadIdx.y;
    
    int tile_w = 5;

    int Row = by * tile_w + ty;
    int Col = bx * tile_w + tx;
    
    if((Row > 0) && (Row < ysize-1)&&(Col > 0) && (Col < xsize-1))
    {
        int offset = Row*xsize + Col;
        int magnitude, sum1, sum2;

        sum1 =  input_pic[ xsize * (Row-1) + Col+1 ] -     input_pic[ xsize*(Row-1) + Col-1 ] 
          +     input_pic[ xsize * (Row)   + Col+1 ] -     input_pic[ xsize*(Row)   + Col-1 ]
          +     input_pic[ xsize * (Row+1) + Col+1 ] -     input_pic[ xsize*(Row+1) + Col-1 ];

        sum2 = input_pic[ xsize * (Row-1) + Col-1 ] +     input_pic[ xsize * (Row-1) + Col ]  + input_pic[ xsize * (Row-1) + Col+1 ]
              - input_pic[xsize * (Row+1) + Col-1 ] -     input_pic[ xsize * (Row+1) + Col ] - input_pic[ xsize * (Row+1) + Col+1 ];

        magnitude =  sum1*sum1 + sum2*sum2;

        if (magnitude > thresh)
          result[offset] = 255;
        else 
          result[offset] = 0;
    }

}


__global__ void gpuPrewittWITHSm(int xsize, int ysize,int thresh, unsigned int *input_pic, int *result)
{
    const int tile_w = 5;
    
    __shared__ int blk_share[tile_w+2][tile_w+2];
    
    int by = blockIdx.x;  int bx = blockIdx.y;
    int ty = threadIdx.x; int tx = threadIdx.y;

    int Row = by * tile_w + ty;
    int Col = bx * tile_w + tx;
    
    blk_share[ty+1][tx+1] = input_pic[ xsize * (Row) + Col ];
    
    if((ty == 0) && (Row > 0))
    {
        blk_share[ty][tx+1] = input_pic[ xsize * (Row-1) + Col ];
    }
    else if((ty == tile_w-1) && (Row < ysize-1))
    {
        blk_share[ty+2][tx+1] = input_pic[ xsize * (Row+1) + Col ];
    }
    
    if((tx==0)&&(Col>0))
    {
        blk_share[ty+1][tx] = input_pic[ xsize * (Row) + Col-1 ];
    }
    else if((tx == tile_w-1) && (Col < xsize-1))
    {
        blk_share[ty+1][tx+2] = input_pic[ xsize * (Row) + Col+1 ];
    }
    
    if((tx==1) && (ty==1)&&(Col-2>=0)&&(Row-2>=0))
    {
        blk_share[0][0] = input_pic[ xsize * (Row-2) + Col-2 ];
    }
    else if((tx==tile_w-2) && (ty==tile_w-2)&&(Row+2 <=ysize-1) && (Col+2 <= xsize-1))
    {
        blk_share[tile_w+1][tile_w+1] = input_pic[ xsize * (Row+2) + Col+2 ];
    }
    else if((tx ==tile_w-2) && (ty==1)&&(Row-2>=0)&&(Col+2 <= xsize-1))
    {
        blk_share[0][tile_w+1] = input_pic[ xsize * (Row-2) + Col+2 ];
    }
    else if((tx ==1) && (ty==tile_w-2)&&(Row+2 <=ysize-1)&&(Col-2>=0))
    {
        blk_share[tile_w+1][0] = input_pic[ xsize * (Row+2) + Col-2 ];
    }
    
    __syncthreads();
    
    
    if((Row > 0) && (Row < ysize-1)&&(Col > 0) && (Col < xsize-1))
    {
        int offset = Row*xsize + Col;
        int magnitude, sum1, sum2;
        
        ty++;
        tx++;
        
       {
          sum1 =  blk_share[ty-1][tx+1] -     blk_share[ty-1][tx-1]  
            +     blk_share[ty][tx+1]   -     blk_share[ty][tx-1]
            +     blk_share[ty+1][tx+1] -     blk_share[ty+1][tx-1];

          sum2 =  blk_share[ty-1][tx-1] +     blk_share[ty-1][tx]  + blk_share[ty-1][tx+1]
                - blk_share[ty+1][tx-1] -     blk_share[ty+1][tx]  - blk_share[ty+1][tx+1];
        }

        magnitude =  sum1*sum1 + sum2*sum2;

        if (magnitude > thresh)
          result[offset] = 255;
        else 
          result[offset] = 0;
    }
    __syncthreads();

}


void GPUPrewittHandler(int xsize, int ysize,int thresh, unsigned int *input_pic, int *output)
{
    int size = xsize * ysize * sizeof(unsigned int);
    
    unsigned int *pic;
    int *result;
    int tile_w = 5;
    
    cudaMalloc(&pic, size);
    cudaMemcpy(pic, input_pic, size, cudaMemcpyHostToDevice);
    
    cudaMalloc(&result, size);
    
    dim3 DimGrid(ceil(double(ysize)/tile_w), ceil(double(xsize)/tile_w), 1);
    dim3 DimBlock(5, 5, 1);

    
//    gpuPrewittWithoutSm<<<DimGrid, DimBlock>>>(xsize, ysize, thresh, pic, result);
    gpuPrewittWITHSm<<<DimGrid, DimBlock>>>(xsize, ysize, thresh, pic, result);
    
    
    cudaMemcpy(output, result, size, cudaMemcpyDeviceToHost);
    
    // Free device matrices
    cudaFree(pic); cudaFree(result);

    
}


int main( int argc, char **argv )
{

  int thresh = DEFAULT_THRESHOLD;
  char *filename;
  filename = strdup( DEFAULT_FILENAME);
  
  if (argc > 1) {
    if (argc == 3)  { // filename AND threshold
      filename = strdup( argv[1]);
       thresh = atoi( argv[2] );
    }
    if (argc == 2) { // default file but specified threshhold
      
      thresh = atoi( argv[1] );
    }

    fprintf(stderr, "file %s    threshold %d\n", filename, thresh); 
  }


  int xsize, ysize, maxval;
  unsigned int *pic = read_ppm( filename, &xsize, &ysize, &maxval ); 


  int numbytes =  xsize * ysize *  sizeof( int );//3 * sizeof( int );
  int *result = (int *) malloc( numbytes );
  if (!result) { 
    fprintf(stderr, "prewitt() unable to malloc %d bytes\n", numbytes);
    exit(-1); // fail
  }

  int i, j, magnitude, sum1, sum2; 
  int *out = result;

  for (int col=0; col<ysize; col++) {
    for (int row=0; row<xsize; row++) { 
      *out++ = 0; 
    }
  }

  // take time snap before Perwit
  high_resolution_clock::time_point t1 = high_resolution_clock::now(); 
 //HostPrewitt(xsize, ysize,thresh, pic, result);
  GPUPrewittHandler(xsize, ysize, thresh, pic, result);
  // take time snap after Perwit
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  // print the Time taken to Multiply two Matrices 
  auto duration = duration_cast<microseconds>(t2 - t1).count();
  cout << "Perwit Time CPU(us):" << duration << "\n";
    
  write_ppm( "result.ppm", xsize, ysize, 255, result);

  fprintf(stderr, "prewitt done\n"); 

}