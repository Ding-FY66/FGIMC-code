clear all
clc
addpath(genpath('./'));

%------导入数据集---------%
Dataname = 'bbcsport4vbigRnSp';
percentDel = 0.3;
Datafold = [Dataname,'_percentDel_',num2str(percentDel),'.mat'];
load(Dataname);
load(Datafold);

%----------参数（需要找最优参数）----------%
l1 = 0.01;    %lambda1
l2 = 10;     %lambda2
l3 = 0.001;     %lambda2
f = 1;
r = 6;   %r = 6取得最优值


t = tic();

T = table( [],[],[],[],[],[],[],[],[],[],[],[], 'VariableNames', { 'f','l1','l2','l3','ACC', 'NMI', 'Purity','F-score','Precision','Recall','ARI','Entropy'});
idx = 1;

for f = 1:10
    ind_folds = folds{f};
    load(Dataname);
    truthF =  truth;   % 真实类标
    clear truth
    numInst = length(truthF);
    num_cluster = length(unique(truthF)); %unique找出truthF中的唯一元素
    numview = length(X);


    for iv = 1:length(X)
        X1 = X{iv};     % 此处X1的维度为d*n
        X1 = NormalizeFea(X1,0);
        ind_0 = find(ind_folds(:,iv) == 0);
        X1(:,ind_0) = [];          % 去除缺失视角
        % -------------- 图初始化 ~S----------------- %
        options = [];
        options.NeighborMode = 'KNN';
        options.k = 11;
        options.WeightMode = 'Binary';       % Binary
        So{iv} = constructW(X1',options);
        G = diag(ind_folds(:,iv));
        G(:,ind_0) = [];
        Sor{iv} = G*So{iv}*G';
    end
    clear X X1 ind_0 G So

    max_iter = 50;
    % % ---------------- 初始化 α ------------- %
    alpha = ones(length(Sor),1)/length(Sor);
    alpha_r = alpha.^r;

    %-----------------初始化 B--------------%
    B = rand(length(Sor),length(Sor));   % 初始化所有视角的重构稀疏权重为1
    B = B-diag(diag(B)); % 去除自表示的影响
    for m = 1:length(Sor)
        indx = [1:length(Sor)];
        indx(m) = [];
        B(indx',m) = (ProjSimplex(B(indx',m)'))'; % 将B的每一列按照列和归一化
    end

    %----------------初始化 U和D--------------%
    U = zeros(numInst,numInst);  % n * n    相似性图
    for iv = 1:length(Sor)
        U = U + Sor{iv};
    end
    U = U./length(Sor);

    D = {};
    for i = 1:numview
        D{i} = U;            %D初始化为U
    end

    %--------------初始化F--------------%
    linshi_U = (U+U')/2;
    LU = diag(sum(linshi_U))-linshi_U;
    [F, ~, ev]=eig1(LU, num_cluster, 0);


    [U,F,obj] = Main(Sor,alpha,B,U,D,F,r,num_cluster,numInst,numview,l1,l2,l3,max_iter,ind_folds);

    new_F = F;
    norm_mat = repmat(sqrt(sum(new_F.*new_F,2)),1,size(new_F,2));
    for i = 1:size(norm_mat,1)
        if (norm_mat(i,1)==0)
            norm_mat(i,:) = 1;
        end
    end
    new_F = new_F./norm_mat;  %哈达玛除法
    pre_labels    = kmeans(real(new_F),num_cluster,'emptyaction','singleton','replicates',20,'display','off'); %kmeans算法

    result_cluster = Clustering8Measure(truthF, pre_labels)*100      %用真实标签和计算的标签分析聚类效果

    T(idx,:) = {f,l1,l2,l3,result_cluster(1),result_cluster(2),result_cluster(3),result_cluster(4),result_cluster(5),result_cluster(6),result_cluster(7),result_cluster(8)};
    idx = idx+1;
    writetable(T, 'data.csv');

end
toc(t)
