%$ \dot{C}(t) = +0.1 * C(t) - 0.02 * C(t) Z(t) $
%$ \dot{Z}(t) = -0.3 * Z(t) + 0.01 * C(t)Z(t) $

clear all; close all; clc

f = @(t,Y) [+0.1 * Y(1) - 0.02 * Y(1) * Y(2); -0.3 * Y(2) + 0.01 * Y(1) * Y(2)];

y1 = linspace(-10,70,20);
y2 = linspace(-10,20,20);

% creates two matrices one for all the x-values on the grid, and one for
% all the y-values on the grid. Note that x and y are matrices of the same
% size and shape, in this case 20 rows and 20 columns
[x,y] = meshgrid(y1,y2);
size(x)
size(y)

u = zeros(size(x));
v = zeros(size(x));

% we can use a single loop over each element to compute the derivatives at
% each point (y1, y2)
t=0; % we want the derivatives at each point at t=0, i.e. the starting time
for i = 1:numel(x)
    Yprime = f(t,[x(i); y(i)]);
    u(i) = Yprime(1);
    v(i) = Yprime(2);
end

quiver(x,y,u,v,'r'); figure(gcf)
xlabel('y_1')
ylabel('y_2')
axis tight equal;

hold on
t = [-10:0.5:70];
y1 = 5 * ones([1, 161]);
y2 = ones([1, 161]) - 1;
plot(t,y1,t,y2,0,0,'o',30,5,'o')
x1 = 0;x2 = 30;
plot([x1,x1],[-10,20],[x2,x2],[-10,20]);
hold off

hold on
%for y10 = [26 28 30 32 34]
%  for y20 = [1 3 5 7 9]   
%for y10 = [40]
%  for y20 = [9]   
%    [ts,ys] = ode45(f,[0:0.5:200],[y10;y20]);
%    plot(ys(:,1),ys(:,2))
%    plot(ys(1,1),ys(1,2),'bo') % starting point
%    plot(ys(end,1),ys(end,2),'ks') % ending point
%  end
%end
%hold off