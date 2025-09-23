all:
	mkdir -p output
	g++ wormhole.cpp -o wormhole -O3 -std=c++17 -ljpeg -lpng -lX11
