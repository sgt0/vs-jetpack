# vs-jetpack

[![Coverage Status](https://coveralls.io/repos/github/Jaded-Encoding-Thaumaturgy/vs-jetpack/badge.svg?branch=main)](https://coveralls.io/github/Jaded-Encoding-Thaumaturgy/vs-jetpack?branch=main)


Full suite of filters, wrappers, and helper functions for filtering video using VapourSynth

`vs-jetpack` provides a collection of Python modules for filtering video using VapourSynth.
These include modules for scaling, masking, denoising, debanding, dehaloing, deinterlacing,
and antialiasing, as well as general utility functions.

For support you can check out the [JET Discord server](https://discord.gg/XTpc6Fa9eB). <br><br>

## How to install

Install `vsjetpack` with the following command:

```sh
pip install vsjetpack
```

Or if you want the latest git version, install it with this command:

```sh
pip install git+https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack.git
```

Note that `vsjetpack` only provides Python functions,
many of them wrapping or combining existing plugins.
You will need to install these plugins separately,
for example using [vsrepo](https://github.com/vapoursynth/vsrepo).
