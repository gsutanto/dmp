% Copyright (C) 2013 Quan Wang <wangq10@rpi.edu>,
% Signal Analysis and Machine Perception Laboratory,
% Department of Electrical, Computer, and Systems Engineering,
% Rensselaer Polytechnic Institute, Troy, NY 12180, USA

% Modified by Giovanni Sutanto <gsutanto@usc.edu>, 
% to also store the optimal (minimum cost) ancestry relationship type 
% of each cell in D, D(i+1,j+1), be it:
% [1] insertion if ancestor is D(i,j+1)
% [2] deletion  if ancestor is D(i+1,j)
% [3] match     if ancestor is D(i,j)

% dynamic time warping of two signals

function [d, D, ancestry_type, correspondences]=dtw(s,t,w)
% s: signal 1, size is ns*k, row for time, colume for channel 
% t: signal 2, size is nt*k, row for time, colume for channel 
% w: window parameter
%      if s(i) is matched with t(j) then |i-j|<=w
% d: resulting distance

if nargin<3
    w=Inf;
end

ns=size(s,1);
nt=size(t,1);
if size(s,2)~=size(t,2)
    error('Error in dtw(): the dimensions of the two input signals do not match.');
end
w=max(w, abs(ns-nt)); % adapt window size

%% initialization
D=zeros(ns+1,nt+1)+Inf; % cost cache matrix
D(1,1)=0;

ancestry_type   = -ones(ns+1,nt+1);
correspondences = cell(0);

%% begin dynamic programming
for i=1:ns
    for j=max(i-w,1):min(i+w,nt)
        oost=norm(s(i,:)-t(j,:));
        [opt_val, opt_idx]  = min( [D(i,j+1), D(i+1,j), D(i,j)] );
        D(i+1,j+1)=oost+opt_val;
        ancestry_type(i+1,j+1)  = opt_idx;
    end
end
d=D(ns+1,nt+1);

%% traverse back ancestry path
i = ns;
j = nt;
while ((i > 0) && (j > 0))
    if (ancestry_type(i+1,j+1) == 1)
        % ancestry == 1: ancestor is 1 cell up (insertion)
        i   = i - 1;
    elseif (ancestry_type(i+1,j+1) == 2)
        % ancestry == 2: ancestor is 1 cell to the left (deletion)
        j   = j - 1;
    elseif (ancestry_type(i+1,j+1) == 3)
        % ancestry == 3: ancestor is 1 cell to the up-left diagonal (match)
        if ((i~=1) || (j~=1))
            correspondences{size(correspondences,1)+1,1}    = [i,j];
        end
        i   = i - 1;
        j   = j - 1;
    else
        msg = 'ERROR: ancestry_type should be either 1, 2, or 3!';
        error(msg);
    end
end