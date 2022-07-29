%{
ref.to 
    https://web.stanford.edu/~boyd/papers/admm/quadprog/quadprog.html
    https://zhuanlan.zhihu.com/p/286235184
%}
%primal problem
G = 2 * [1 0;0 1];
g = [-2;-4];
Aeq = [2 3];
beq = 5;
low = [0.5;0];
up = [2;3];

admm_x = ADMM_QP()
quadprog_x = quadprog(G,g,[],[],Aeq,beq,low,up)

%ref. ppt P.10
function [finalx] = ADMM_QP()
%{
  Solve above QP problem using ADMM, decompose primal problem into two 
  problem: min(x-1)^2 and min(y-2)^2, then alternating optimization
        min (x-1)^2 + (y-2)^2
        s.t 0.5 <= x <= 2
            0 <= y <= 3
            2x+3y=5
%}
    x = 1;
    y = 1;
    lambda = 1;
    maxIter = 20;
    rho = 1.5;
    
    [H_xx_sym, g_x_sym] = getHession_g('x');
    [H_yy_sym, g_y_sym] = getHession_g('y');
    lb = [0.5;0];
    ub = [2;3];
    i = 1;
    while i <= maxIter
        H_xx = eval(H_xx_sym);
        g_x = eval(g_x_sym);
        x = quadprog(H_xx,g_x,[],[],[],[],lb(1),ub(1),[]);
        H_yy = eval(H_yy_sym);
        g_y = eval(g_y_sym);
        y = quadprog(H_yy,g_y,[],[],[],[],lb(2),ub(2),[]);
        lambda = lambda + rho*(2*x+3*y-5);
        i = i + 1;
        finalx = [x;y];
    end
end

function [H,g] = getHession_g(fn)
%{
    fn : function
    H && g is standard QP
        minimize 0.5*x'*H*x + g'*x
%}
    syms x y lambda rho; 
    if strcmp(fn,'x')
        %固定y，默认x为符号变量
        f = (x-1)^2 + lambda*(2*x + 3*y -5) + rho/2*(2*x + 3*y -5)^2;
        % x^2项系数
        H = hessian(f,x);
        % x项系数，直接将上式展开取x前的系数即可
        g = (2*lambda + (rho*(12*y - 20))/2 - 2);
    elseif strcmp(fn,'y')
        %固定x，默认y为符号变量
        f = (y-2)^2 + lambda*(2*x + 3*y -5) + rho/2*(2*x + 3*y -5)^2;
        H = hessian(f,y);
        g = (3*lambda + (rho*(12*x - 30))/2 - 4);
    end
end