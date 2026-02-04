% fidname='./origin_data/t1_mse.img.fid';
%fidname='./origin_data/t2_fse.img.fid';
fidname='./origin_data/PD-SE.img.fid';
figure('NumberTitle', 'off', 'Name', fidname);

[tau,Data,pp,section] = read_fid(fidname);
for i = 1:section
    
    subplot(section,3,i);
    FreqData= imrotate(abs(Data(:,:,i)),90);
    imshow(FreqData,[]);
    title('K空间');
    
    subplot(section,3,i+3);
    ReImage=ifftshift(ifft2(Data(:,:,i)));              %二维傅里叶反变换
    img = abs(ReImage);                                 
    img = gray_trans(img,[0,255]);                       %灰度拉伸
    img = imrotate(img,90);
    img = fliplr(img);
    img = imresize(img,[256,256]);
    imshow(img,[]);
    title('重建结果');
end
