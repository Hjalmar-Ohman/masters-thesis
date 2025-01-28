## Setup
python
requirements.txt

### Poppler
Go to [this page](https://github.com/oschwartz10612/poppler-windows/releases/tag/v24.08.0-0) and download the binary of your choice. In this example we will download and use poppler-0.68.0_x86.

Extract the archive file poppler-0.68.0_x86.7z into C:\Program Files. Thus, the directory structure should look something like this:

C:
    └ Program Files
        └ poppler-0.68.0_x86
            └ bin
            └ include
            └ lib
            └ share
Add C:\Program Files\poppler-0.68.0_x86\bin to your system PATH by doing the following: Click on the Windows start button, search for Edit the system environment variables, click on Environment Variables..., under System variables, look for and double-click on PATH, click on New, then add C:\Users\Program Files\poppler-0.68.0_x86\bin, click OK.

If you are using a terminal to execute poppler (e.g. running pdf2image in command line), you may need to reopen your terminal for poppler to work.

Done!