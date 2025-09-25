# Morris-Thorne Wormhole Ray-Traced Simulation

Run:

```
make
./wormhole
```

Combine images into video:

```
ffmpeg -framerate 24 -i output/4k/wormhole_%d.png -c:v libx264 -crf 1 -profile:v high -pix_fmt yuv420p -preset veryslow output.mp4
```