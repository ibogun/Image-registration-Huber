%% Image stitching
% Here we use our method on real data for image stitching. Image stitching
% is the problem of combining two different images of the same scene into
% one. Since we are able to find transformation from one image to another
% we can transform one of the to the same reference frame.

% get images. Resize to more managable sizes ( for faster processing)
I1=imresize(imread('1.jpg'),0.2);
I2=imresize(imread('2.jpg'),0.2);

[n,m,~]=size(I1);

newN=1000;
D1=zeros(newN,newN,3,'uint8');
D2=zeros(newN,newN,3,'uint8');

mask=zeros(newN,newN);

delta_x=100;
delta_y=200;

mask(delta_y+1:delta_y+n,delta_x+1:delta_x+m)=1;
[n,m,~]=size(I1);

D1(delta_y+1:delta_y+n,delta_x+1:delta_x+m,:)=I1;
D2(delta_y+1:delta_y+n,delta_x+1:delta_x+m,:)=I2;

toPlot=1;

if (toPlot)
    
    h2=figure;
    
    subplot(1,2,1);
    imshow(I1);
    axis off;
    title('Image 1 ');
    
    subplot(1,2,2);
    imshow(I2);
    axis off;
    
    title('Image 2');
end


I = single(rgb2gray(D1));
[fa, da] = vl_sift(I) ;
[fb, db] = vl_sift(single(rgb2gray(D2))) ;

[matches, scores] = vl_ubcmatch(da, db) ;

d1=fa(1:2,matches(1,:));
d2=fb(1:2,matches(2,:));

if (toPlot)
    %plot matches
    match_plot(D1,D2,d1(:,1:30)',d2(:,1:30)');
    title('Feature matches between images');
    axis off;
end

d1=[ones(1,size(d1,2));d1];

d1=d1(:,1:90);
d2=d2(:,1:90);
d1=d1';
d2=d2';

A=[d1,zeros(size(d1));zeros(size(d1)),d1];
b=[d2(:,1);d2(:,2)];

[x,w,~,optval]=solveRobustLeastSquares(A,b,1);

% get transformation
T=[x(2:3)' x(1); x(5:6)' x(4)];
T_inv=cv.invertAffineTransform(T);

I_rect=cv.warpAffine(D2,T_inv,'BorderValue',[255 255 255],'DSize',[newN,newN]);

newMask=round(cv.warpAffine(mask,T_inv,'DSize',[newN,newN]));

% get combined mask
combined=uint8(mask.*newMask);

mask=logical(mask);
newMask=logical(newMask);

only_im1=xor(mask,combined);
only_im2=xor(newMask,combined);

f=@(im,mask) im.*repmat(uint8(mask),[1,1,3]);

blend1=I_rect.*repmat(combined,[1,1,3]);
blend2=D1.*repmat(combined,[1,1,3]);

figure;
blended=blendMode(blend1,blend2, 'Normal', 1, 1);
%imshow(blended);

final_image=(double(f(D1,only_im1))/255)+(blended)+(double(f(I_rect,only_im2))/255);

imshow(final_image);
title('Stitched image');
