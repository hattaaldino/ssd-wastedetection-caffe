@echo off

set trainvalloc=Images\TRAIN\

set testloc=Images\TEST\

rem create imageset txt
type nul > trainval.txt
type nul > text.txt

echo Start data recording...

rem iterate trainval data
for /r "%trainvalloc%" %%v in ("*.jpg") do (

echo %%~nv >> trainval.txt

)

echo Trainval data has been recorded

rem iterate text data
for /r "%testloc%" %%t in ("*.jpg") do (

echo %%~nt >> text.txt

)

echo Test data has been recorded

pause