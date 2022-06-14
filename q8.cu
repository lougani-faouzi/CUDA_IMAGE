#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <string.h>
#include "FreeImage.h"
#define WIDTH 1920
#define HEIGHT 1024
#define BPP 24 // Since we're outputting three 8 bit RGB values

#define BLOCK_WIDTH 32

using namespace std;

__global__ void blur_filter(unsigned int* source, unsigned int* destination, unsigned width, unsigned height)
{
  int colonne,ligne;
  colonne  = threadIdx.x + blockDim.x * blockIdx.x;
  ligne  = threadIdx.y + blockDim.y * blockIdx.y;

if((ligne < (height-1))&&(colonne < (width-1))&&(ligne > 0)&&(colonne > 0))
{
  // for each pixel color, get the mean value of the pixel and its 4 immediate neighbors (5 total)
  
  destination[((ligne * width + colonne) * 3) + 0] = (source[((ligne * width + colonne + 1) * 3) + 0]
  + source[((ligne * width + colonne - 1) * 3) + 0] + source[(((ligne + 1) * width + colonne) * 3) + 0]
  + source[(((ligne - 1) * width + colonne) * 3) + 0] + source[((ligne * width + colonne) * 3) + 0])/5;
  
  
  destination[((ligne * width + colonne) * 3) + 1] = (source[((ligne * width + colonne + 1) * 3) + 1]
  + source[((ligne * width + colonne - 1) * 3) + 1] + source[(((ligne + 1) * width + colonne) * 3) + 1]
  + source[(((ligne - 1) * width + colonne) * 3) + 1] + source[((ligne * width + colonne) * 3) + 1])/5;
  
  
  destination[((ligne * width + colonne) * 3) + 2] = (source[((ligne * width + colonne + 1) * 3) + 2]
  + source[((ligne * width + colonne - 1) * 3) + 2] + source[(((ligne + 1) * width + colonne) * 3) + 2]
  + source[(((ligne - 1) * width + colonne) * 3) + 2] + source[((ligne * width + colonne) * 3) + 2])/5;
}
 
}

int main (int argc , char** argv)
{
  FreeImage_Initialise();
  const char *PathName = "img.jpg";
  const char *PathDest = "new_img.png";
  // load and decode a regular file
  FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(PathName);

  FIBITMAP* bitmap = FreeImage_Load(FIF_JPEG, PathName, 0);

  if(! bitmap )
    exit( 1 ); //WTF?! We can't even allocate images ? Die !

  unsigned width  = FreeImage_GetWidth(bitmap);
  unsigned height = FreeImage_GetHeight(bitmap);
  unsigned pitch  = FreeImage_GetPitch(bitmap);

  fprintf(stderr, "Processing Image of size %d x %d\n", width, height);

  unsigned int *img = (unsigned int*) malloc(sizeof(unsigned int) * 3 * width * height);

  unsigned int* cuda_source;
  unsigned int* cuda_destination;
  
  // allocation 
  cudaMalloc(&cuda_source, sizeof(unsigned int) * 3 * width * height); 
  cudaMalloc(&cuda_destination, sizeof(unsigned int) * 3 * width * height); 
  
  // convert image to array
  BYTE *bits = (BYTE*)FreeImage_GetBits(bitmap);
  for ( int y =0; y<height; y++)
  {
    BYTE *pixel = (BYTE*)bits;
    for ( int x =0; x<width; x++)
    {
      int idx = ((y * width) + x) * 3;
      img[idx + 0] = pixel[FI_RGBA_RED];
      img[idx + 1] = pixel[FI_RGBA_GREEN];
      img[idx + 2] = pixel[FI_RGBA_BLUE];
      pixel += 3;
    }
    // next line
    bits += pitch;
  }

  cudaMemcpy(cuda_source, img, 3 * width * height * sizeof(unsigned int), cudaMemcpyHostToDevice);

  int nbBlocksx = width / BLOCK_WIDTH;
  if(width % BLOCK_WIDTH) nbBlocksx++;

  int nbBlocksy = height / BLOCK_WIDTH;
  if(height % BLOCK_WIDTH) nbBlocksy++;

  fprintf(stderr, "(%d, %d) blocks of size (%d, %d)\n", nbBlocksx, nbBlocksy, BLOCK_WIDTH, BLOCK_WIDTH);

  dim3 gridSize(nbBlocksx, nbBlocksy);
  dim3 blockSize(BLOCK_WIDTH, BLOCK_WIDTH);
  
  // use blur filter kernel
  blur_filter<<<gridSize, blockSize>>>(cuda_source, cuda_destination, width, height);

  cudaMemcpy(img, cuda_destination, 3 * width * height * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  // save image
  bits = (BYTE*)FreeImage_GetBits(bitmap);
  for ( int y =0; y<height; y++)
  {
    BYTE *pixel = (BYTE*)bits;
    for ( int x =0; x<width; x++)
    {
      RGBQUAD newcolor;

      int idx = ((y * width) + x) * 3;
      newcolor.rgbRed = img[idx + 0];
      newcolor.rgbGreen = img[idx + 1];
      newcolor.rgbBlue = img[idx + 2];

      if(!FreeImage_SetPixelColor(bitmap, x, y, &newcolor))
      { fprintf(stderr, "(%d, %d) Fail...\n", x, y); }

      pixel+=3;
    }
   bits += pitch;
  }

  if( FreeImage_Save (FIF_PNG, bitmap , PathDest , 0 ))
    cout << "Image successfully saved ! " << endl ;
  FreeImage_DeInitialise(); //Cleanup !

  free(img);
  cudaFree(cuda_destination);
  cudaFree(cuda_source);
}
