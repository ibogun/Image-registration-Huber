%% Image registration using robust least squares
% This script shows how feature-based image registration can be solved
% using robust least squares. Using SIFT features for matching we propose
% alternative to RANSAC method for finding affine transformation between
% images. Results are evaluated in two settings: using synthetic
% transformation and for image stitching.

%% Dependencies
% Author:   Ivan Bogun
%
% Date:     April 6, 2014
%
% Required:
%
% * Matlab mex functions for opencv: <http://www.cs.stonybrook.edu/~kyamagu/mexopencv/ mexopencv>
% * VLfeat library  : <http://www.vlfeat.org/ vlfeat>
% * Source code: 
% Note: in theory vlfeat functions can be replaced with analogs from
% mexopencv (SIFT feature extraction and matching). After everything is
% installed add mexopencv and vlfeat to the PATH.


%% Prepare the data
% Affine Transformation
% \begin{equation}
% [x,y]\rightarrow [D_{11}x+ D_{12}y+D_{13},D_{21}x+ D_{22}y+D_{23}]
% \end{equation}

% transformation
D=[0.7 0.4 5;0 0.64 0];

I1=imread('linux.jpeg');
I2=cv.warpAffine(I1,D,'BorderValue',[255 255 255]);

I = single(rgb2gray(I1));
h2=figure;

subplot(1,2,1);
imshow(I1);
axis off;
title('Original ');

subplot(1,2,2);
imshow(I2);
axis off;

title('Transformed');

%% Feature extraction
% Extract SIFT features from both images.

[fa, da] = vl_sift(I) ;
[fb, db] = vl_sift(single(rgb2gray(I2))) ;

% plot couple of features
toPlot=1;
if (toPlot)
    figure;
    image(I1);
    axis off;
    [f,d] = vl_sift(I) ;
    
    title('SIFT features on the image');
    perm = randperm(size(f,2)) ;
    sel = perm(1:10) ;
    h1 = vl_plotframe(f(:,sel)) ;
    h2 = vl_plotframe(f(:,sel)) ;
    set(h1,'color','k','linewidth',3) ;
    set(h2,'color','y','linewidth',2) ;
    
    h3 = vl_plotsiftdescriptor(d(:,sel),f(:,sel)) ;
    set(h3,'color','g') ;
    
end

%% Feature matching
% Ideally we would like to have as low as possible false matches,
% as they will be hard-to-detect outliers in the later fitting step. In
% order to decrease such number of outliers two features are matched if
% second best match is within some distance from the best one. This method
% was introduced [1] by Lowe (2004).
[matches, scores] = vl_ubcmatch(da, db) ;

d1=fa(1:2,matches(1,:));
d2=fb(1:2,matches(2,:));

if (toPlot)
    %plot matches
    match_plot(I1,I2,d1',d2');
    title('Feature matches between images');
    axis off;
end

%% Robust least squares
% Inevitably previous step will create spurious matches which, if fitting
% is performed with regular least-squares, will significantly influence
% final result. RANSAC is the classical method used to overcome it. In
% RANSAC we iteratively take subset of matches, perform fitting and evaluate
% fitted parameters. If everything goes well _right_ subset of correct
% matches will be chosen which will lead to a good fit. Here, we propose
% to use robust least squares with Huber penalty function. Let $A \in
% \mathbf{R}^{m \times n}, b \in \mathbf{R}^m$ - be locations of matched
% features in image 1 and 2 respectively. In regular least-squares we seek
% to find transformation $x \in \mathbf{R}^n$ which  minimizes objective
% function : $f(x)= \sum_{i=1}^m \Vert a_i ^T x-b \Vert^2$. Huber robust loss function
% defined with parameter $M$ which is quadratic in the region $\{x: |x|\leq M\}$
% and linear elsewhere. Here it's definition
% \[\phi_M (x) = \left\{
%   \begin{array}{lr}
%     \Vert x\Vert^2 & :\Vert x\Vert\leq M\\
%     M(2\Vert x \Vert-M) & :\Vert x \Vert>M
%   \end{array}
% \right.
% \]

% define huber function with M=1
huber=@(x) le(abs(x),1).*abs(x).^2+ge(abs(x),1).*(2*abs(x)-1);
x=linspace(-4,4,1000);
huber_x=huber(x);
quadratic_x=x.^2;

figure;
plot(x,huber_x,'LineWidth',2,'Color','g');
hold on;
plot(x,quadratic_x,'LineWidth',2,'Color','b');
hold on;

plot([-1 -1],[0 16],'Color','r','LineStyle','--');
plot([1 1],[0 16],'Color','r','LineStyle','--');
legend('Huber loss','Quadratic loss');
title('Different loss functions');
hold off


%% Solving robust least squares
% With huber loss function new objective has the form:
% \begin{equation}
% \min._x \sum_{i=1}^m \phi_M(\Vert a_i ^T x-b \Vert^2)
% \end{equation}
% It can be shown that the problem above is equivalent to the following
% convex optimization problem [2] (page 195):
% \begin{align}
% \min._{x,w} &\:\:\sum_{i=1}^m \frac{\Vert a_i^T x -b \Vert^2}{(1+w_i)}+M^2 \sum_{i=1}^m w_i  \\
% \text{s.t.} & \:\: w_i\geq 0 \:\:\:\:\: \:\:\:\:\: \forall i=1,...,m
% \end{align}
% which we solve using interior point primal dual algorithm [2].

d1=[ones(1,size(d1,2));d1];
d1=d1';
d2=d2';

A=[d1,zeros(size(d1));zeros(size(d1)),d1];
b=[d2(:,1);d2(:,2)];

% robust least squares parameter
M=1;
param.debug=1;
[x,w,~,optval]=solveRobustLeastSquares(A,b,M,param);


%% Reconstruction
% Reconstruct the image using newly found transformation and plot the
% results.
T=[x(2:3)' x(1); x(5:6)' x(4)];

T_inv=cv.invertAffineTransform(T);
I_rect=cv.warpAffine(I2,T_inv,'BorderValue',[255 255 255]);
figure;
subplot(1,2,1);
imshow(I1);
axis off;
title('Original');
subplot(1,2,2);
imshow(I_rect);
title('Reconstructed');
axis off;

H = padarray(2,[2 2]) - fspecial('gaussian' ,[5 5],2);
sharpened = imfilter(I_rect,H);

figure;
subplot(1,2,1);
imshow(I1);
axis off;
title('Original');
subplot(1,2,2);
imshow(sharpened);
title('Reconstructed and sharpened');
axis off;

fprintf('Reconstruction error: %3.3f \n',norm(double(I1(:))-double(I_rect(:))));

%% Image stitching
% Here we use our method on real data for image stitching. Image stitching
% is the problem of combining two different images of the same scene into
% one. Since we are able to find transformation from one image to another
% we can transform both images onto same reference frame.

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
    figure;
    imshow(I1);
    axis off;
    title('Image 1 ');
    
    figure;
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
    match_plot(D1,D2,d1(:,1:20)',d2(:,1:20)');
    title('Feature matches between images');
    axis off;
end

d1=[ones(1,size(d1,2));d1];

% Use only part of the features because of efficiency reasons.
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
axis off;
title('Stitched image');



%% References
% [1] Lowe, David G. "Distinctive image features from scale-invariant keypoints." International journal of computer vision 60.2 (2004): 91-110.
%
% [2] Boyd, Stephen P., and Lieven Vandenberghe. Convex optimization. Cambridge university press, 2004.