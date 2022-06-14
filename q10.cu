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

__global__ void sobel_filter(unsigned int* source, unsigned int* destination, unsigned width, unsigned height)
{
  int colonne,ligne;
  colonne  = threadIdx.x + blockDim.x * blockIdx.x;
  ligne  = threadIdx.y + blockDim.y * blockIdx.y;
  
  int gx, gy, g;
  
if((ligne < (height-1))&&(colonne < (width-1))&&(ligne > 0)&&(colonne > 0))
{
  // apply two convolution to the image:
  
  // gx filter -1  0  1
  //           -2  0  2
  //           -1  0  1
  gx =
  (-1 * source[(((ligne - 1) * width + colonne - 1) * 3) + 0]) + (source[(((ligne - 1) * width + colonne + 1) * 3) + 0]) + 
  (-2 * source[((ligne * width + colonne - 1) * 3) + 0]) + (2 * source[((ligne * width + colonne + 1) * 3) + 0]) + 
  (-1 * source[(((ligne + 1) * width + colonne - 1) * 3) + 0]) + (source[(((ligne + 1) * width + colonne + 1) * 3) + 0]);
  
  
  // gy filter  1  2  1
  //            0  0  0
  //           -1 -2 -1
  
  gy =
  (source[(((ligne - 1) * width + colonne - 1) * 3) + 0]) + (2 * source[(((ligne - 1) * width + colonne) * 3) + 0]) + 
  (source[(((ligne - 1) * width + colonne + 1) * 3) + 0]) + (-1 * source[(((ligne + 1) * width + colonne - 1) * 3) + 0]) + 
  (-2 * source[(((ligne + 1) * width + colonne) * 3) + 0]) + (-1 * source[(((ligne + 1) * width + colonne + 1) * 3) + 0]);
  
  // then compute sqaured root of gx**2 + gy**2
  g = sqrtf((gx*gx)+(gy*gy));
  
  // same gray value for all colors
  destination[((ligne * width + colonne) * 3) + 0] = g;
  destination[((ligne * width + colonne) * 3) + 1] = g;
  destination[((ligne * width + colonne) * 3) + 2] = g;
  
}
 
}

int main (int argc , char** argv)
{
  FreeImage_Initialise();
  // use gray scale image not rgb
  const char *PathName = "gray_img.png";
  const char *PathDest = "new_img.png";
  // load and decode a regular file
  FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(PathName);

  FIBITMAP* bitmap = FreeImage_Load(fif, PathName, 0);

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
  
  // call sobel filter kernel, wihch is for edge detection (works for gray scale images)
  sobel_filter<<<gridSize, blockSize>>>(cuda_source, cuda_destination, width, height);

  cudaMemcpy(img, cuda_destination, 3 * width * height * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  
  
  // save image into disk
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
