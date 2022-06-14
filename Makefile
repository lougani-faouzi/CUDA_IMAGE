all:
	nvcc -I${HOME}/softs/FreeImage/include q6.cu -L${HOME}/softs/FreeImage/lib/ -lfreeimage -o q6
	nvcc -I${HOME}/softs/FreeImage/include q7.cu -L${HOME}/softs/FreeImage/lib/ -lfreeimage -o q7
	nvcc -I${HOME}/softs/FreeImage/include q8.cu -L${HOME}/softs/FreeImage/lib/ -lfreeimage -o q8
	nvcc -I${HOME}/softs/FreeImage/include q9.cu -L${HOME}/softs/FreeImage/lib/ -lfreeimage -o q9
	nvcc -I${HOME}/softs/FreeImage/include q10.cu -L${HOME}/softs/FreeImage/lib/ -lfreeimage -o q10
	nvcc -I${HOME}/softs/FreeImage/include q11.cu -L${HOME}/softs/FreeImage/lib/ -lfreeimage -o q11
	nvcc -I${HOME}/softs/FreeImage/include q12.cu -L${HOME}/softs/FreeImage/lib/ -lfreeimage -o q12
	nvcc -I${HOME}/softs/FreeImage/include q14.cu -L${HOME}/softs/FreeImage/lib/ -lfreeimage -o q14

clean:
	rm -f *.o q6 q7 q8 q9 q10 q11 q12 q14
