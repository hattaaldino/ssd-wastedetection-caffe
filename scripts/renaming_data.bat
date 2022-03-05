@echo off

rem define data location
set loc=%~1

rem define name template
set name=%~2

rem define initial
set init=%3

rem define file extension
set ext=%~4

rem define counter
set count=%init%

rem temp file for store assigned file names
type nul > names.tmp

rem for each file in folder loc
for %%i in ("%loc%*.*") do (

call :rename "%loc%"="%%~ni"="%name%"="%%~xi"

)

del names.tmp

rem measure how much files are changed
set /a affected = count-init

echo Total affected file^: %affected%

exit /b

rem rename function
:rename

findstr %~2 names.tmp > nul 2> nul

if %ERRORLEVEL% equ 0 exit /b

set /a count+=1

rem create new name for current file
set newname=%~3%count%.%ext%

rem store newname into temporary storage
echo %newname% >> names.tmp

ren "%~1%~2%~4" "%newname%" 2> nul

if %ERRORLEVEL% equ 1 (

goto :rename

)

exit /b

