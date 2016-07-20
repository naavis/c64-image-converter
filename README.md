Commodore 64 Color Converter
============================

This Python script takes an image file and makes it compatible with the
Commodore 64 standard bitmap mode (high resolution mode).
Images in the standard bitmap mode are 320x200 pixels in resolution
and divided into 8x8 pixel blocks. Only two colors are allowed in each
block. The Commodore 64 color palette has 16 fixed colors.

![Before and after](images/before_and_after.png?raw=true "Before and after color mangling")
Photo before and after conversion shown above.

Usage
-----

```bash
python c64convert.py filename.png
```

Currently only the preview image is given. Sane data output is still unimplemented.

To-do
-----

- Output color indices in some sane format.
- Handle multicolor bitmap mode.
- Dithering for gradients.

Requirements
------------

- Python 3.x
- Numpy
- Scikit-image
- Matplotlib

License
-------
The project is licensed under the MIT license, reproduced below.

The MIT License (MIT)

Copyright (c) 2016 Samuli Vuorinen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
