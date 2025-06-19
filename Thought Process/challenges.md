Documentation of Challenges 

1. Spatial Gaussian Regularization: I need to learn how to implement anisotropic Gaussian kernels that respect scene geometry, preventing floating artifacts while maintaining detail.

2. View-Dependent Appearance: I'm going to have to model specular reflections and lighting using spherical Gaussians, so that I allow realistic material properties across viewpoints. I need to figure this out ASAP!

3. Semantic-Aware Splatting: I willneed to use CLIP embeddings to guide Gaussian placement, (this will ensure semantically meaningful 3D structure rather than pure photometric optimization!!).