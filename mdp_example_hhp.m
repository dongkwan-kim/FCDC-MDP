%{
Consider two simple propagations. (The number of news M = 2)
News A: 1->2->3
News B: 1->2,3
(e,f) for A in t=1,2,3
: (1, *), (2, *) ,(3, *)

(e,f) for B in t=1,2,3
: (1, *), (3, *) ,(3, *)

Budget K = 1;
%}


% State : Status of checked or not. (c1, ... ,cM) where ci is 0 or 1.
% Action : (a1, ... ,an) where ai is 0 or 1 and \sum ai = K.
% transition probability is computed by
% p_t(j|,s_t,,a) = 1 if j = s_t + a , 0 otherwise.

T = 3; % upper limit of time 
K = 1;
M = 2;
d = 0:(2^M-1);
S = de2bi(d,[],2,'left-msb'); % b: state of states
%b(i, :) == (i-1) with a form of a binary vector.

expsr = nan(M, T); % status matrix of exposure, each row denotes the news, and t-th column mean the cumulated number of exposure at time t.
flags = nan(M, T); % status matrix of exposure, each row denotes the news, and t-th column mean the cumulated number of fake-flag at time t.

%% for the simple simulation,
expsr = [1,2,3;1,3,3];
flags = [0,0,1;1,2,2];

nS = size(S,1); % == 2^M : number of the whole states.
Actions = nan(nchoosek(M,K), M); % set of actions.
nA = size(Actions,1); % == nchoosek(M,K); % number of the whole actions.
P = zeros(nS,nS, nA); %P(SxSxA) = transition probability matrix 

tmp = 0;
for i = 1:nS
    if sum(S(i,:)) == K
        tmp = tmp +1;
        Actions(tmp, :) = S(i,:);
    end
end
    
for j = 1:nS
    for a = 1:nA
        nextS = S(j,:) + Actions(a,:);
        if ~ prod(nextS < 2) % if the actions is unfeasible
            P(j,j, a) = 1; % let the state hold
%             disp([j,a])
        else
            P(j, bi2de(nextS,'left-msb') + 1, a) = 1;
        end
    end
end
% P

%   R(SxA) = reward matrix
% The reward does not depend on S but only A
% Here a in in {1, 2, ... , nchoosek(M, K)}.
RewardMatrix = nan(nS, nA, T);
RewardMatrix(:,:,T) = zeros(nS, nA);
for t = 1:T-1
    for j = 1:nA
        a = Actions(j,:);
        checkIdx = 1:M;
        checkIdx = checkIdx(logical(a));
        for k = checkIdx
            RewardMatrix(:,j,t) = (expsr(k, t+1) - expsr(k,t))*flags(k,t)*ones(nS,1);
        end
    end
end
discount = 1;
N = T;
h = zeros(nS,1);
[V, policy, cpu_time] = mdp_finite_horizon_time(P, RewardMatrix, discount, N, h)
