function [tau,Data,pp,section]=read_fid(FileName)
SW=20;
fid=fopen(FileName,'r');
if(fid<=0)
    errordlg('Failure to open the file !','Error','on');
else
    FileVersion=fread(fid,1,'int32');
    Section1Size=fread(fid,1,'int32');
    Section2Size=fread(fid,1,'int32');
    Section3Size=fread(fid,1,'int32');
    Section4Size=fread(fid,1,'int32');
    Section5Size=fread(fid,1,'int32');
    Position=Section1Size+Section2Size+Section3Size+Section4Size;%计算偏移字节数；
    TempData=fread(fid,Position,'int8');
    Dimension1=fread(fid,1,'int32');
    Dimension2=fread(fid,1,'int32');
    Dimension3=fread(fid,1,'int32');
    Dimension4=fread(fid,1,'int32');
    DataReal=zeros(Dimension1,Dimension2,Dimension3,Dimension4);
    DataImaginary=zeros(Dimension1,Dimension2,Dimension3,Dimension4);
    for l=1:Dimension4
        for k=1:Dimension3
            for j=1:Dimension2
                Data=fread(fid,[2,Dimension1],'float32')';
                DataReal(:,j,k,l)=Data(:,1);
                DataImaginary(:,j,k,l)=Data(:,2);
            end
        end
    end
    fclose(fid);
end
section = Dimension3;
tau = (0:1:Dimension1-1)'./SW;
Data = DataReal+1i*DataImaginary;
pp = 0;
