#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include "FreeImage.h"

#define BPP 24 // Since we're outputting three 8 bit RGB values

using namespace std;

__global__ void Pop_art_Warhol(int * d_img, int *img, unsigned width, unsigned height)
{

  int colonne = blockIdx.x * blockDim.x + threadIdx.x;
  int ligne = blockIdx.y * blockDim.y + threadIdx.y;

  int indx = ((ligne * width) + colonne) * 3;
 
  // top right becomes green satured
  if ((ligne > height/2) && (colonne  > width/2))
  {
    d_img[indx] = img[indx];
    d_img[indx+1] = 255;
    d_img[indx+2] = img[indx+2];
  }
  
  // bottom right becomes blue satured
  if ((ligne < height/2) && (colonne > width/2))
  {
    d_img[indx] = img[indx];
    d_img[indx+1] = img[indx+1];
    d_img[indx+2] = 255;
  } 
 
  // bottom left becomes red satured
  if ((ligne < height/2) && (colonne <= width/2))
  {
    d_img[indx] = 255;
    d_img[indx+1] = img[indx+1];
    d_img[indx+2] = img[indx+2];
  }
  
  // leave top left as is

}
int main(int argc, char** argv)
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

  int size = sizeof(unsigned int) * 3 * width * height;

  unsigned int *img = (unsigned int*) malloc(size);

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

  int *d_img,*d_img1;
  cudaMalloc((void**)&d_img,size);
  cudaMemcpy(d_img, img,size,cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_img1,size);
  cudaMemcpy(d_img1, img,size,cudaMemcpyHostToDevice);

  dim3 dimGrid (128,72,1);
  dim3 dimBlock (30,30,1);
  
  // use kernel to divide image  into 2*2 blocks, each satured with a specific color
  Pop_art_Warhol<<<dimGrid, dimBlock>>>(d_img,d_img1, width, height);

  cudaMemcpy(img,d_img,size,cudaMemcpyDeviceToHost);

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
    // next line
    bits += pitch;
  }

  if( FreeImage_Save (FIF_PNG, bitmap , PathDest , 0 ))
    printf("Image successfully saved !\n");
  FreeImage_DeInitialise(); //Cleanup !

  cudaFree(d_img);
  cudaFree(d_img1);

  free(img);

  return 0;
}
