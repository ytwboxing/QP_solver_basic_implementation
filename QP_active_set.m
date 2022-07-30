%{
ref.to 
    https://github.com/nhorswill/Active-Set-Method-for-Quadratic-Programming/blob/master/active_set_QP.m
    materials/active_set.pdf
%}
G=[1 0;0 1];
c=[-2;-5];
% Aeq=[3 1];
% beq=10;
Aeq=[];
beq=[];
A=[1 -2;-1 -2;-1 2;1 0;0 1;Aeq;-Aeq];
b=[-2;-6;-2;0;0;beq;-beq];
%init solve and working set
x_0=[0;0];
W_0=[];

[x,W,q] = active_set_QP(G,c,A,b,x_0,W_0);
my_cost_value = q;
active_set_x = x

quadprog_x = quadprog(G,c,-A,-b,Aeq,beq)

function [x,W,q] = active_set_QP(G,c,A,b,x_0,W_0)
%{
    The following algorithm provides a solution to quadratic programs of
    the following form:
    
    Minimize q(x) = 0.5*x'*G*x+c'*x
    s.t. a'*x=b
         aa'*x>=bb
%}

q=[];
x=[x_0];
W = [W_0];
x_k = x_0;
W_k = W_0;
k = 0;
dims = size(A);%row * col
indices = 1:dims(1);%1 : row
stop = 0;
    while stop==0
       %返回在indices存在而在W_k中不存在的元素
       W_c = setdiff(indices,W_k);
       k=k+1;
       % equation (6)
       g_k = G*x_k+c;
       dimens=size(A(W_k,:));
       if isempty(W_k)
           p = quadprog(G,g_k);
       else
           p = quadprog(G,g_k,[],[],A(W_k,:),zeros(dimens(1),1));
       end
       for i=1:length(p)
           if abs(p(i))<1e-7
               p(i)=0;
           end
       end
       % Case 1
       if any(p ~= 0)
           change=A(W_c,:)*p;
           for k=1:length(change)
               if abs(change(k))<1e-7
                   change(k)=0;
               end
           end
           rtindex = find(change < 0);
           newindex = W_c(rtindex);
           small = (b(newindex,:)-A(newindex,:)*x_k)./(A(newindex,:)*p);
           rt=min(small);
           ak = min([1,rt]);
           x_k = x_k+ak*p;
       % Case 1.1
           if ak==1
               continue;
           end
       % Case 1.2
           if (ak~=1)
               j = find(small==rt);
               j1 = W_c(j(1));
               W_k = [W_k, j1];
           end
       end
       % Case 2
       if all(p==0)
           lambda = linsolve(A(W_k,:)',g_k);
       % Case 2.1
           if all(lambda >=0)
               stop=1;
           end
       % Case 2.2
           if any(lambda<0)
               j = find(lambda<0);
               j1=j(1);
               x_k=x_k;
               W_k(j1) = [];
           end
       end
       % save and return
       q_k = x_k'*G*x_k/2+c'*x_k;
       x = [x_k];
       if isempty(W_k)
           W_final=NaN;
       else
           W_final=W_k;
       end
       W = [W_final];
       q = [q_k];
    end
end