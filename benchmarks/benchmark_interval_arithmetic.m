clear;
clc;

% load data
load('data.mat');

% evaluation
addition_eval(addition_data);
subtraction_eval(subtraction_data);
multiplication_eval(multiplication_data);
division_eval(division_data);
power_eval(power_data);
absolute_eval(absolute_data);
left_matrix_multiplication_eval(left_matrix_multiplication_data);
right_matrix_multiplication_eval(right_matrix_multiplication_data);
exponential_eval(exponential_data);
log_eval(log_data);
sqrt_eval(sqrt_data);
asin_eval(arcsin_data);
acos_eval(arccos_data);
atan_eval(arctan_data);
sinh_eval(sinh_data);
cosh_eval(cosh_data);
tanh_eval(tanh_data);
asinh_eval(arcsinh_data);
acosh_eval(arccosh_data);
atanh_eval(arctanh_data);
sin_eval(sin_data);
cos_eval(cos_data);
tan_eval(tan_data);
% cot_eval(cot_data); % no cot supported in CORA

% --------------------------------------------------------------------------------------------------
% FUNCTIONALITY
% function cot_eval(data)
%     sz = size(data.I_inf);
%     runs = sz(2);
% 
%     I = interval(data.I_inf,data.I_sup);
% 
%     % detect time cost ++++++++++++++++++++++
%     t1 = datetime;
%     res = cot(I);
%     t2 = datetime;
%     % detect time cost ++++++++++++++++++++++
% 
%     t_seconds = seconds(t2-t1)/runs;
% 
%     % compute epsilon
% 
%     diff_inf = abs(res.inf-data.res_inf);
%     diff_sup = abs(res.sup-data.res_sup);
%     epsilon = max(max(diff_inf,diff_sup)./(data.res_sup-data.res_inf));
% 
%     % disp
%     disp('------------------------------------')
%     disp('cot_eval '+string(runs)+' Runs AVG. cost: '+string(t_seconds));
%     disp('epsilon: '+string(epsilon));
%     disp('------------------------------------')
% end

function tan_eval(data)
    sz = size(data.I_inf);
    runs = sz(2);

    I = interval(data.I_inf,data.I_sup);

    % detect time cost ++++++++++++++++++++++
    t1 = datetime;
    res = tan(I);
    t2 = datetime;
    % detect time cost ++++++++++++++++++++++

    t_seconds = seconds(t2-t1)/runs;

    % compute epsilon

    diff_inf = abs(res.inf-data.res_inf);
    diff_sup = abs(res.sup-data.res_sup);
    epsilon = max(max(diff_inf,diff_sup)./(data.res_sup-data.res_inf));

    % disp
    disp('------------------------------------')
    disp('tan_eval '+string(runs)+' Runs AVG. cost: '+string(t_seconds));
    disp('epsilon: '+string(epsilon));
    disp('------------------------------------')
end

function cos_eval(data)
    sz = size(data.I_inf);
    runs = sz(2);

    I = interval(data.I_inf,data.I_sup);

    % detect time cost ++++++++++++++++++++++
    t1 = datetime;
    res = cos(I);
    t2 = datetime;
    % detect time cost ++++++++++++++++++++++

    t_seconds = seconds(t2-t1)/runs;

    % compute epsilon

    diff_inf = abs(res.inf-data.res_inf);
    diff_sup = abs(res.sup-data.res_sup);
    epsilon = max(max(diff_inf,diff_sup)./(data.res_sup-data.res_inf));

    % disp
    disp('------------------------------------')
    disp('cos_eval '+string(runs)+' Runs AVG. cost: '+string(t_seconds));
    disp('epsilon: '+string(epsilon));
    disp('------------------------------------')
end

function sin_eval(data)
    sz = size(data.I_inf);
    runs = sz(2);

    I = interval(data.I_inf,data.I_sup);

    % detect time cost ++++++++++++++++++++++
    t1 = datetime;
    res = sin(I);
    t2 = datetime;
    % detect time cost ++++++++++++++++++++++

    t_seconds = seconds(t2-t1)/runs;

    % compute epsilon

    diff_inf = abs(res.inf-data.res_inf);
    diff_sup = abs(res.sup-data.res_sup);
    epsilon = max(max(diff_inf,diff_sup)./(data.res_sup-data.res_inf));

    % disp
    disp('------------------------------------')
    disp('sin_eval '+string(runs)+' Runs AVG. cost: '+string(t_seconds));
    disp('epsilon: '+string(epsilon));
    disp('------------------------------------')
end


function atanh_eval(data)
    sz = size(data.I_inf);
    runs = sz(2);

    I = interval(data.I_inf,data.I_sup);

    % detect time cost ++++++++++++++++++++++
    t1 = datetime;
    res = atanh(I);
    t2 = datetime;
    % detect time cost ++++++++++++++++++++++

    t_seconds = seconds(t2-t1)/runs;

    % compute epsilon

    diff_inf = abs(res.inf-data.res_inf);
    diff_sup = abs(res.sup-data.res_sup);
    epsilon = max(max(diff_inf,diff_sup)./(data.res_sup-data.res_inf));

    % disp
    disp('------------------------------------')
    disp('atan_eval '+string(runs)+' Runs AVG. cost: '+string(t_seconds));
    disp('epsilon: '+string(epsilon));
    disp('------------------------------------')
end

function acosh_eval(data)
    sz = size(data.I_inf);
    runs = sz(2);

    I = interval(data.I_inf,data.I_sup);

    % detect time cost ++++++++++++++++++++++
    t1 = datetime;
    res = acosh(I);
    t2 = datetime;
    % detect time cost ++++++++++++++++++++++

    t_seconds = seconds(t2-t1)/runs;

    % compute epsilon

    diff_inf = abs(res.inf-data.res_inf);
    diff_sup = abs(res.sup-data.res_sup);
    epsilon = max(max(diff_inf,diff_sup)./(data.res_sup-data.res_inf));

    % disp
    disp('------------------------------------')
    disp('acosh_eval '+string(runs)+' Runs AVG. cost: '+string(t_seconds));
    disp('epsilon: '+string(epsilon));
    disp('------------------------------------')
end

function asinh_eval(data)
    sz = size(data.I_inf);
    runs = sz(2);

    I = interval(data.I_inf,data.I_sup);

    % detect time cost ++++++++++++++++++++++
    t1 = datetime;
    res = asinh(I);
    t2 = datetime;
    % detect time cost ++++++++++++++++++++++

    t_seconds = seconds(t2-t1)/runs;

    % compute epsilon

    diff_inf = abs(res.inf-data.res_inf);
    diff_sup = abs(res.sup-data.res_sup);
    epsilon = max(max(diff_inf,diff_sup)./(data.res_sup-data.res_inf));

    % disp
    disp('------------------------------------')
    disp('asinh_eval '+string(runs)+' Runs AVG. cost: '+string(t_seconds));
    disp('epsilon: '+string(epsilon));
    disp('------------------------------------')
end

function tanh_eval(data)
    sz = size(data.I_inf);
    runs = sz(2);

    I = interval(data.I_inf,data.I_sup);

    % detect time cost ++++++++++++++++++++++
    t1 = datetime;
    res = tanh(I);
    t2 = datetime;
    % detect time cost ++++++++++++++++++++++

    t_seconds = seconds(t2-t1)/runs;

    % compute epsilon

    diff_inf = abs(res.inf-data.res_inf);
    diff_sup = abs(res.sup-data.res_sup);
    epsilon = max(max(diff_inf,diff_sup)./(data.res_sup-data.res_inf));

    % disp
    disp('------------------------------------')
    disp('tanh_eval '+string(runs)+' Runs AVG. cost: '+string(t_seconds));
    disp('epsilon: '+string(epsilon));
    disp('------------------------------------')
end

function cosh_eval(data)
    sz = size(data.I_inf);
    runs = sz(2);

    I = interval(data.I_inf,data.I_sup);

    % detect time cost ++++++++++++++++++++++
    t1 = datetime;
    res = cosh(I);
    t2 = datetime;
    % detect time cost ++++++++++++++++++++++

    t_seconds = seconds(t2-t1)/runs;

    % compute epsilon

    diff_inf = abs(res.inf-data.res_inf);
    diff_sup = abs(res.sup-data.res_sup);
    epsilon = max(max(diff_inf,diff_sup)./(data.res_sup-data.res_inf));

    % disp
    disp('------------------------------------')
    disp('cosh_eval '+string(runs)+' Runs AVG. cost: '+string(t_seconds));
    disp('epsilon: '+string(epsilon));
    disp('------------------------------------')
end


function sinh_eval(data)
    sz = size(data.I_inf);
    runs = sz(2);

    I = interval(data.I_inf,data.I_sup);

    % detect time cost ++++++++++++++++++++++
    t1 = datetime;
    res = sinh(I);
    t2 = datetime;
    % detect time cost ++++++++++++++++++++++

    t_seconds = seconds(t2-t1)/runs;

    % compute epsilon

    diff_inf = abs(res.inf-data.res_inf);
    diff_sup = abs(res.sup-data.res_sup);
    epsilon = max(max(diff_inf,diff_sup)./(data.res_sup-data.res_inf));

    % disp
    disp('------------------------------------')
    disp('sinh_eval '+string(runs)+' Runs AVG. cost: '+string(t_seconds));
    disp('epsilon: '+string(epsilon));
    disp('------------------------------------')
end

function atan_eval(data)
    sz = size(data.I_inf);
    runs = sz(2);

    I = interval(data.I_inf,data.I_sup);

    % detect time cost ++++++++++++++++++++++
    t1 = datetime;
    res = atan(I);
    t2 = datetime;
    % detect time cost ++++++++++++++++++++++

    t_seconds = seconds(t2-t1)/runs;

    % compute epsilon

    diff_inf = abs(res.inf-data.res_inf);
    diff_sup = abs(res.sup-data.res_sup);
    epsilon = max(max(diff_inf,diff_sup)./(data.res_sup-data.res_inf));

    % disp
    disp('------------------------------------')
    disp('atan_eval '+string(runs)+' Runs AVG. cost: '+string(t_seconds));
    disp('epsilon: '+string(epsilon));
    disp('------------------------------------')
end

function acos_eval(data)
    sz = size(data.I_inf);
    runs = sz(2);

    I = interval(data.I_inf,data.I_sup);

    % detect time cost ++++++++++++++++++++++
    t1 = datetime;
    res = acos(I);
    t2 = datetime;
    % detect time cost ++++++++++++++++++++++

    t_seconds = seconds(t2-t1)/runs;

    % compute epsilon

    diff_inf = abs(res.inf-data.res_inf);
    diff_sup = abs(res.sup-data.res_sup);
    epsilon = max(max(diff_inf,diff_sup)./(data.res_sup-data.res_inf));

    % disp
    disp('------------------------------------')
    disp('acos_eval '+string(runs)+' Runs AVG. cost: '+string(t_seconds));
    disp('epsilon: '+string(epsilon));
    disp('------------------------------------')
end

function asin_eval(data)
    sz = size(data.I_inf);
    runs = sz(2);

    I = interval(data.I_inf,data.I_sup);

    % detect time cost ++++++++++++++++++++++
    t1 = datetime;
    res = asin(I);
    t2 = datetime;
    % detect time cost ++++++++++++++++++++++

    t_seconds = seconds(t2-t1)/runs;

    % compute epsilon

    diff_inf = abs(res.inf-data.res_inf);
    diff_sup = abs(res.sup-data.res_sup);
    epsilon = max(max(diff_inf,diff_sup)./(data.res_sup-data.res_inf));

    % disp
    disp('------------------------------------')
    disp('asin_eval '+string(runs)+' Runs AVG. cost: '+string(t_seconds));
    disp('epsilon: '+string(epsilon));
    disp('------------------------------------')
end

function sqrt_eval(data)
    sz = size(data.I_inf);
    runs = sz(2);

    I = interval(data.I_inf,data.I_sup);

    % detect time cost ++++++++++++++++++++++
    t1 = datetime;
    res = sqrt(I);
    t2 = datetime;
    % detect time cost ++++++++++++++++++++++

    t_seconds = seconds(t2-t1)/runs;

    % compute epsilon

    diff_inf = abs(res.inf-data.res_inf);
    diff_sup = abs(res.sup-data.res_sup);
    epsilon = max(max(diff_inf,diff_sup)./(data.res_sup-data.res_inf));

    % disp
    disp('------------------------------------')
    disp('sqrt_eval '+string(runs)+' Runs AVG. cost: '+string(t_seconds));
    disp('epsilon: '+string(epsilon));
    disp('------------------------------------')
end

function log_eval(data)
    sz = size(data.I_inf);
    runs = sz(2);

    I = interval(data.I_inf,data.I_sup);

    % detect time cost ++++++++++++++++++++++
    t1 = datetime;
    res = log(I);
    t2 = datetime;
    % detect time cost ++++++++++++++++++++++

    t_seconds = seconds(t2-t1)/runs;

    % compute epsilon

    diff_inf = abs(res.inf-data.res_inf);
    diff_sup = abs(res.sup-data.res_sup);
    epsilon = max(max(diff_inf,diff_sup)./(data.res_sup-data.res_inf));

    % disp
    disp('------------------------------------')
    disp('log_eval '+string(runs)+' Runs AVG. cost: '+string(t_seconds));
    disp('epsilon: '+string(epsilon));
    disp('------------------------------------')
end

function exponential_eval(data)
    sz = size(data.I_inf);
    runs = sz(2);

    I = interval(data.I_inf,data.I_sup);

    % detect time cost ++++++++++++++++++++++
    t1 = datetime;
    res = exp(I);
    t2 = datetime;
    % detect time cost ++++++++++++++++++++++

    t_seconds = seconds(t2-t1)/runs;

    % compute epsilon

    diff_inf = abs(res.inf-data.res_inf);
    diff_sup = abs(res.sup-data.res_sup);
    epsilon = max(max(diff_inf,diff_sup)./(data.res_sup-data.res_inf));

    % disp
    disp('------------------------------------')
    disp('exp_eval '+string(runs)+' Runs AVG. cost: '+string(t_seconds));
    disp('epsilon: '+string(epsilon));
    disp('------------------------------------')
end

function right_matrix_multiplication_eval(data)
    sz = size(data.I_inf);
    runs = sz(1);

    I = cell(1,runs);
    for i = 1:runs
        this_inf = squeeze(data.I_inf(i,:,:));
        this_sup = squeeze(data.I_sup(i,:,:));
        this_interval = interval(this_inf, this_sup);
        I{i}=this_interval;
    end

    m = data.m;

    res={1,runs};

    % detect time cost ++++++++++++++++++++++
    t1 = datetime;
    for i =1:runs
        res{i} = m*I{i};
    end
    t2 = datetime;
    % detect time cost ++++++++++++++++++++++

    t_seconds = seconds(t2-t1)/runs;

    % compute epsilon

    res_inf = {};
    res_sup = {};

    for i = 1:runs
        res_inf{i}=res{i}.inf;
        res_sup{i}=res{i}.sup;
    end

    res_inf = cat(3,res_inf{:});
    res_inf = permute(res_inf,[3,1,2]);
    res_sup = cat(3,res_sup{:});
    res_sup = permute(res_sup,[3,1,2]);

    diff_inf = abs(res_inf-data.res_inf);
    diff_sup = abs(res_sup-data.res_sup);
    diff_max = max(diff_inf,diff_sup);
    epsilon_mat= diff_max./(data.res_sup-data.res_inf);
    epsilon = max(epsilon_mat,[],'all');

    % disp
    disp('------------------------------------')
    disp('right_matrix_multiplication_eval '+string(runs)+' Runs AVG. cost: '+string(t_seconds));
    disp('epsilon: '+string(epsilon));
    disp('------------------------------------')
end


function left_matrix_multiplication_eval(data)
    sz = size(data.I_inf);
    runs = sz(1);

    I = cell(1,runs);
    for i = 1:runs
        this_inf = squeeze(data.I_inf(i,:,:));
        this_sup = squeeze(data.I_sup(i,:,:));
        this_interval = interval(this_inf, this_sup);
        I{i}=this_interval;
    end

    m = data.m;

    res={1,runs};

    % detect time cost ++++++++++++++++++++++
    t1 = datetime;
    for i =1:runs
        res{i} = I{i}*m;
    end
    t2 = datetime;
    % detect time cost ++++++++++++++++++++++

    t_seconds = seconds(t2-t1)/runs;

    % compute epsilon

    res_inf = {};
    res_sup = {};

    for i = 1:runs
        res_inf{i}=res{i}.inf;
        res_sup{i}=res{i}.sup;
    end

    res_inf = cat(3,res_inf{:});
    res_inf = permute(res_inf,[3,1,2]);
    res_sup = cat(3,res_sup{:});
    res_sup = permute(res_sup,[3,1,2]);

    diff_inf = abs(res_inf-data.res_inf);
    diff_sup = abs(res_sup-data.res_sup);
    diff_max = max(diff_inf,diff_sup);
    epsilon_mat= diff_max./(data.res_sup-data.res_inf);
    epsilon = max(epsilon_mat,[],'all');

    % disp
    disp('------------------------------------')
    disp('left_matrix_multiplication_eval '+string(runs)+' Runs AVG. cost: '+string(t_seconds));
    disp('epsilon: '+string(epsilon));
    disp('------------------------------------')
end

function absolute_eval(data)
    sz = size(data.I_inf);
    runs = sz(2);

    I = interval(data.I_inf,data.I_sup);

    % detect time cost ++++++++++++++++++++++
    t1 = datetime;
    res = abs(I);
    t2 = datetime;
    % detect time cost ++++++++++++++++++++++

    t_seconds = seconds(t2-t1)/runs;

    % compute epsilon

    diff_inf = abs(res.inf-data.res_inf);
    diff_sup = abs(res.sup-data.res_sup);
    epsilon = max(max(diff_inf,diff_sup)./(data.res_sup-data.res_inf));

    % disp
    disp('------------------------------------')
    disp('absolute_eval '+string(runs)+' Runs AVG. cost: '+string(t_seconds));
    disp('epsilon: '+string(epsilon));
    disp('------------------------------------')
end

function power_eval(data)
    sz = size(data.I_inf);
    runs = sz(2);

    I = interval(data.I_inf,data.I_sup);
    NUM = double(data.NUM);

    res = [];

    % detect time cost ++++++++++++++++++++++
    t1 = datetime;
    for i = 1:runs
        this_res = I(i)^NUM;
        res=[res,this_res];
    end
    t2 = datetime;
    % detect time cost ++++++++++++++++++++++

    t_seconds = seconds(t2-t1)/runs;

    % compute epsilon

    diff_inf = abs(res.inf-data.res_inf);
    diff_sup = abs(res.sup-data.res_sup);
    epsilon = max(max(diff_inf,diff_sup)./(data.res_sup-data.res_inf));

    % disp
    disp('------------------------------------')
    disp('division_eval '+string(runs)+' Runs AVG. cost: '+string(t_seconds));
    disp('epsilon: '+string(epsilon));
    disp('------------------------------------')
end

function division_eval(data)
    sz = size(data.I_inf);
    runs = sz(2);

    I = interval(data.I_inf,data.I_sup);


    res = [];

    % detect time cost ++++++++++++++++++++++
    t1 = datetime;
    for i = 1:runs
        this_res = I(i)/I(i);
        res=[res,this_res];
    end
    t2 = datetime;
    % detect time cost ++++++++++++++++++++++

    t_seconds = seconds(t2-t1)/runs;

    % compute epsilon

    diff_inf = abs(res.inf-data.res_inf);
    diff_sup = abs(res.sup-data.res_sup);
    epsilon = max(max(diff_inf,diff_sup)./(data.res_sup-data.res_inf));

    % disp
    disp('------------------------------------')
    disp('division_eval '+string(runs)+' Runs AVG. cost: '+string(t_seconds));
    disp('epsilon: '+string(epsilon));
    disp('------------------------------------')
end

function multiplication_eval(data)
    sz = size(data.I_inf);
    runs = sz(2);

    I = interval(data.I_inf,data.I_sup);

    res = [];

    % detect time cost ++++++++++++++++++++++
    t1 = datetime;
    for i = 1:runs
        this_res = I(i)*I(i);
        res=[res,this_res];
    end
    t2 = datetime;
    % detect time cost ++++++++++++++++++++++

    t_seconds = seconds(t2-t1)/runs;

    % compute epsilon

    diff_inf = abs(res.inf-data.res_inf);
    diff_sup = abs(res.sup-data.res_sup);
    epsilon = max(max(diff_inf,diff_sup)./(data.res_sup-data.res_inf));

    % disp
    disp('------------------------------------')
    disp('multiplication_eval '+string(runs)+' Runs AVG. cost: '+string(t_seconds));
    disp('epsilon: '+string(epsilon));
    disp('------------------------------------')
end

function subtraction_eval(data)
    sz = size(data.I_inf);
    runs = sz(2);

    I = interval(data.I_inf,data.I_sup);

    % detect time cost ++++++++++++++++++++++
    t1 = datetime;
    res = I-I;
    t2 = datetime;
    % detect time cost ++++++++++++++++++++++

    t_seconds = seconds(t2-t1)/runs;

    % compute epsilon

    diff_inf = abs(res.inf-data.res_inf);
    diff_sup = abs(res.sup-data.res_sup);
    epsilon = max(max(diff_inf,diff_sup)./(data.res_sup-data.res_inf));

    % disp
    disp('------------------------------------')
    disp('subtraction_eval '+string(runs)+' Runs AVG. cost: '+string(t_seconds));
    disp('epsilon: '+string(epsilon));
    disp('------------------------------------')
end


function addition_eval(data)
    sz = size(data.I_inf);
    runs = sz(2);
    I = interval(data.I_inf,data.I_sup);

    % detect time cost ++++++++++++++++++++++
    t1 = datetime;
    res = I+I;
    t2 = datetime;
    % detect time cost ++++++++++++++++++++++

    t_seconds = seconds(t2-t1)/runs;

    % compute epsilon

    diff_inf = abs(res.inf-data.res_inf);
    diff_sup = abs(res.sup-data.res_sup);    
    epsilon = max(max(diff_inf,diff_sup)./(data.res_sup-data.res_inf));

    % disp
    disp('------------------------------------')
    disp('addition_eval '+string(runs)+' Runs AVG. cost: '+string(t_seconds));
    disp('epsilon: '+string(epsilon));
    disp('------------------------------------')
end

