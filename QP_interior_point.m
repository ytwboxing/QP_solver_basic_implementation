%{
ref.to 
    https://github.com/Ashton-Sidhu/InteriorPointMethod/blob/master/run_IPM.m
    materials/interior_point.pdf
    
    内点法的实现方法有很多种，具体解释可参考：https://zhuanlan.zhihu.com/p/165930639
    这里仅实现Predictor-corrector method
    
%}
G=[1 0;0 1];
c=[-2;-5];
A=[1 -2;-1 -2;-1 2;1 0;0 1];
b=[-2;-6;-2;0;0];
quadprog_x = quadprog(G,c,-A,-b)
[z,finalx] = interior_point_QP( G, c, -A, -b );
ipm_x = finalx
function [z,finalx] = interior_point_QP( H, c, A, b )
%{
IPM Solves a quadratic programming problem in the form
    minimize 0.5*x'*H*x + c'*x
    s.t Ax <= b  
IPM uses the prediction and central path method to find
its way to the most optimal solution. 
%}
    [m,n] = size(A);
    x = ones(n,1);
    lamda = ones(m,1);
    s = ones(m,1);
    
    %Calculate residual values and mu
    rd0 = H*x + c + A'*lamda;
    rp0 = s + A*x - b;
    mu = sum(lamda.*s)/m;
    e = ones(m,1);
    rc = diag(s) * diag(lamda)*e;
    i = 0;
    eps = 1e-7;
    maxiter = 300;
    while i <= maxiter && norm(rd0) >= eps && norm(rp0) >= eps && abs(mu) >= eps
        %#############################################
        %predictor step
        %#############################################
        %(2.11)
        big = [H A' zeros(n,m); 
               A zeros(m,m) eye(m); 
               zeros(m,n) diag(s) diag(lamda)];
        r = [-rd0; -rp0; -rc];
        aff = big\r;

        xaff = aff(1:n);
        lamdaaff = aff(n+1:n+m);
        saff = aff(n+m+1:n+m+m);
        
        %Compute alpha_affine
        alphaaff = 1;
        indexfindz = find ( lamdaaff <0) ;
        if (isempty ( indexfindz )==0)
            %(2.19)
            alphaaff = min (alphaaff , min(-lamda (indexfindz ) ./ lamdaaff ( indexfindz ) ) ) ;
        end
        indexfinds = find ( saff  <0) ;
        if(isempty ( indexfinds )==0)
            %(2.20)
            alphaaff = min ( alphaaff , min(-s(indexfinds ) ./ saff(indexfinds) ) ) ;
        end
        %compute mu_affine
        %(2.12)
        muaff = (s + alphaaff .* saff)'*(lamda + alphaaff .* lamdaaff)/m;
        %#############################################
        %compute centering parameter
        %#############################################
        %(2.13)
        cent = (muaff/mu)^3;
        
        %#############################################
        %Corrector and centering step
        %#############################################
        %obtain search direction
        %(2.25)
        imprc = rc + saff.*lamdaaff - cent*mu*e;
        deltars = [-rd0; -rp0; -imprc];
        deltas = big\deltars;
        deltax = deltas(1:n);
        deltalamda = deltas(n+1:n+m);
        deltas = deltas(n+m+1:n+m+m);
        
        %#############################################
        %compute alpha
        %##############################################
        stepsizealpha = 1 ;
        indexfindz = find( deltalamda<0) ;
        if(isempty ( indexfindz)==0)
            stepsizealpha = min ( stepsizealpha , min(-lamda ( indexfindz ) ./ deltalamda(indexfindz ) ) ) ;
        end
        indexfinds = find(deltas<0) ;
        if ( isempty( indexfinds )==0)
            stepsizealpha = min ( stepsizealpha , min(-s ( indexfinds ) ./ deltas( indexfinds ) ) ) ;
        end
        
        %#############################################
        %update(x,lamda,s)
        %#############################################
        x = x + stepsizealpha*deltax;
        lamda = lamda + stepsizealpha*deltalamda;
        s = s + stepsizealpha*deltas;
        
        %#############################################
        %update residuals and complementarity measure
        %#############################################
        rd0 = H*x + c + A'*lamda;
        rp0 = s + A*x -b;        
        rc = diag(s) * diag(lamda)*e; 
        mu = sum(s.*lamda)/m;
        i = i + 1;
    end
    finalx = x;
    z=((0.5*finalx')*(H*finalx))+(c'*finalx);
end