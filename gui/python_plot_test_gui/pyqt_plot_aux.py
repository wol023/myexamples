def check_and_try_cluster(pathfilename,host=[],username=[],password=[],basepath=[],targetpath=[]):
    head=os.path.split(pathfilename)
    path_loc=head[0]
    file_loc=head[1]
    print path_loc
    print file_loc

    homedir=os.getcwd()

    if not os.path.exists(path_loc):
        os.mkdir(path_loc)
        print path_loc+'/', 'is created.'
    else:
        print path_loc+'/', 'already exists.'

    status=0

    currentdirlist=os.listdir(path_loc)
    if file_loc in currentdirlist:
        print file_loc, 'is found in local machine.'
        status = 1
    else:
        print file_loc, 'is NOT found in local machine.. start downloading.'
        if host!=[] and username!=[] and password!=[] and basepath!=[] and targetpath!=[]:
            with pysftp.Connection(host=host, username=username, password=base64.b64decode(password),cnopts=cnopts) as sftp:
                if sftp.exists(basepath):
                    with sftp.cd(basepath):
                        if sftp.exists(targetpath):
                            with sftp.cd(targetpath):
                                    if sftp.exists(path_loc):
                                        with sftp.cd(path_loc):
                                            if sftp.exists(file_loc):
                                                os.chdir(path_loc)
                                                sftp.get(file_loc, preserve_mtime=True,callback=printProgress)
                                                print file_loc,'download completed.'
                                                os.chdir(homedir)
                                                status=2
                                            else:
                                                print file_loc,'is not found in', host
                                                status=-1
                                    else:
                                        print path_loc,'is not found in', host
                                        status=-2

                        else:
                            print targetpath,'is not found in', host
                            status=-3
                else:
                    print basepath,'is not found in', host
                    status=-4

    return status


