 clear all;

% Define mean vectors and covariance matrices for two Gaussians
mu1 = [3 3];
sigma1 = [1.2 -0.4; -0.4 1.2];
mu2 = [6 6];
sigma2 = [1.2 0.4; 0.4 1.2];

% Define the range of x and y values
x = linspace(0, 10, 50);
y = linspace(0, 10, 50);
[X, Y] = meshgrid(x, y);

P = [0.1 0.25 0.5 0.75 0.9];

for i=1:5
    P1 = P(i);
    P2 = 1-P(i);

    pdf1 = mvnpdf([X(:) Y(:)], mu1, sigma1);
    pdf1 = reshape(pdf1, length(y), length(x));
    pdf1 = pdf1*P1;

    pdf2 = mvnpdf([X(:) Y(:)], mu2, sigma2);
    pdf2 = reshape(pdf2, length(y), length(x));
    pdf2 = pdf2*P2;
    
    if i==3
        figure(6);
        contour(X,Y,pdf1, 10, 'g');
        hold on;
        contour(X,Y,pdf2, 10, 'c');
        axis square;
        title('green: W1..........cyan: W2');
        grid on;
    end

    figure(i);
    hold on;
    surf(x, y, pdf1, 'FaceColor', 'magenta', 'EdgeColor','black');
   
    surf(x, y, pdf2, 'FaceColor', 'cyan', 'EdgeColor','black');

    z = -(1.25*X.*Y - 22.5) - log(P2/P1);
    surf(x, y, z, 'FaceColor', 'black', 'EdgeColor','none');

    title(['P1=', num2str(P1), ', P2=', num2str(P2)]);
xlim([min(x) max(x)]);
    ylim([min(y) max(y)]);
    zlim([0 0.16]);
   

    figure(7);
    hold on;
     surf(x, y, z, 'FaceColor', 'black', 'EdgeColor','none');
     xlim([min(x) max(x)]);
    ylim([min(y) max(y)]);
    zlim([0 0.16]);
end

sigma = sigma2

for i=1:5
    P1 = P(i);
    P2 = 1-P(i);

    pdf1 = mvnpdf([X(:) Y(:)], mu1, sigma);
    pdf1 = reshape(pdf1, length(y), length(x));
    pdf1 = pdf1*P1;

    pdf2 = mvnpdf([X(:) Y(:)], mu2, sigma);
    pdf2 = reshape(pdf2, length(y), length(x));
    pdf2 = pdf2*P2;
    
    if i==3
        figure(8);
        contour(X,Y,pdf1, 10, 'g');
        hold on;
        contour(X,Y,pdf2, 10, 'c');
        axis square;
        title('green: W1..........cyan: W2');
        grid on;
    end

    figure(9+i);
    hold on;
    surf(x, y, pdf1, 'FaceColor', 'magenta', 'EdgeColor','black');
   
    surf(x, y, pdf2, 'FaceColor', 'cyan', 'EdgeColor','black');

    z = -(3.75*(X+Y) - 33.75) - log(P2/P1);
    surf(x, y, z, 'FaceColor', 'black', 'EdgeColor','none');

    title(['P1=', num2str(P1), ', P2=', num2str(P2)]);
xlim([min(x) max(x)]);
    ylim([min(y) max(y)]);
    zlim([0 0.16]);
   

    figure(9);
    hold on;
     surf(x, y, z, 'FaceColor', 'black', 'EdgeColor','none');
     xlim([min(x) max(x)]);
    ylim([min(y) max(y)]);
    zlim([0 0.16]);
end