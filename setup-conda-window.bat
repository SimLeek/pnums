CALL conda.bat env create neuralnumbers
CALL conda.bat activate neuralnumbers
CALL conda.bat install pytorch torchvision cudatoolkit=10.0 -c pytorch
CALL conda.bat install -c conda-forge opencv
CALL pip install cvpubsubs matplotlib tox-conda
