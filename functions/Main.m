function [U,F,obj] = Main(Sor_ini,alpha_ini,B_ini,U_ini,D_ini,F_ini,r,num_cluster,numInst,numview,lambda1,lambda2,lambda3,max_iter,ind_folds,f)

So = Sor_ini;
Sor = Sor_ini;
alpha = alpha_ini;
B = B_ini;
U = U_ini;
D = D_ini;
F = F_ini;
alpha_r = alpha.^r;
a = 14;
b = 25;

for iter = 1:max_iter
    if iter == 1
        Sor = Sor_ini;
    end
    A = {};
    Q = zeros(1,num_cluster);
    for i=1:numview
        if f==4
            a = 12; b =27 ;
        end
        if f==3
            a =20;
        elseif f==6 || f==7 || f == 8
            a =8;
        end
        T=D{i};
        T(T<0)=0;
        [T,~]=choose_neighbor_coefficient(T,a,b);
        T=(T+T')/2;
        A{i}=T.*Sor{i};
        Q(i)=1/2/norm(A{i}-U,'fro');
    end
    Rec_error = zeros(1,length(Sor));
    vec_D = [];
    for iv = 1:length(Sor)
        vec_D = [vec_D,(D{iv}(:))];
    end
    for iv = 1:length(Sor)
        W = ones(numInst,numInst);
        ind_0 = find(ind_folds(:,iv) == 0); 
        W(:,ind_0) = 0;
        W(ind_0,:) = 0;
        Rec_error(iv) = norm((1/2)*(U+D{iv}).*W-Sor{iv},'fro')^2 + lambda1*norm(vec_D(:,iv)-vec_D*B(:,iv))^2 + lambda2*Q(iv)*norm(U-A{iv},'fro')^2;
    end
    clear LSv W
    H = bsxfun(@power,Rec_error, 1/(1-r));     
    alpha = bsxfun(@rdivide,H,sum(H));
    alpha_r = alpha.^r;
    clear H
    vec_D = [];
    for iv = 1:length(Sor)
        vec_D = [vec_D,(Sor{iv}(:))];
    end
    for iv = 1:length(Sor)
        indv = [1:length(Sor)];
        indv(iv) = [];
        [B(indv',iv),~] = SimplexRepresentation_acc(vec_D(:,indv), vec_D(:,iv));
    end
    O = {};
    vec_D = [];
    for iv = 1:length(Sor)
        vec_D = [vec_D,(D{iv}(:))];
    end
    for iv = 1:length(Sor)
        W = ones(numInst,numInst);
        ind_0 = find(ind_folds(:,iv) == 0); 
        W(:,ind_0) = 0;
        W(ind_0,:) = 0;
        M_iv = vec_D*B(:,iv);
        M_iv = reshape(M_iv,numInst,numInst);
        O{iv} = (1/2)*U.*W-Sor{iv};
        sum_Y = 0;
        coeef = 0;
        for iv2 = 1:length(Sor)
            if iv2 ~= iv
                Y_iv2 = vec_D(:,iv2)-vec_D*B(:,iv2)+B(iv,iv2)*vec_D(:,iv);
                sum_Y = sum_Y + alpha_r(iv2)*B(iv,iv2)*lambda1*Y_iv2;
                coeef = coeef +  B(iv,iv2)^2*alpha_r(iv2);
                clear Y_iv2
            end
        end
        clear iv2
        matrix_sum_Y = reshape(sum_Y,numInst,numInst);
        clear sum_Y
        Linshi_L = (matrix_sum_Y+alpha_r(iv)*lambda1*M_iv+lambda2*Q(iv)*alpha_r(iv)*U-(1/2)*alpha_r(iv)*W.*O{iv})./((1/4)*alpha_r(iv)*W+lambda2*Q(iv)*alpha_r(iv)+lambda1*alpha_r(iv)+coeef*lambda1);

        for num = 1:numInst
            indnum = [1:numInst];
            indnum(num) = [];
            D{iv}(indnum',num) = (EProjSimplex_new(Linshi_L(indnum',num)'))';
        end
        clear Linshi_L matrix_sum_Y coeef
    end
    clear vec_D
    First_item = zeros(numInst,numInst);
    Sec_item = zeros(numInst,numInst);
    F_item = zeros(numInst,numInst);
    S_item = zeros(numInst,numInst);
    for i = 1:numInst
        for iv = 1:length(Sor)
            W = ones(numInst,numInst);
            ind_0 = find(ind_folds(:,iv) == 0); 
            W(:,ind_0) = 0;
            W(ind_0,:) = 0;
            First_item(:,i) = First_item(:,i) + alpha_r(iv)*(Sor{iv}(:,i)-(1/2)*D{iv}(:,i).*W(:,i));
            Sec_item(:,i) = Sec_item(:,i) + 2*alpha_r(iv)*lambda2*Q(iv)*A{iv}(:,i);
            F_item(:,i) = F_item(:,i) + alpha_r(iv)*W(:,i)/2;
            S_item(:,i) = S_item(:,i) + 2*alpha_r(iv)*lambda2*Q(iv);
        end
    end
    for i = 1:numInst
        all=distance(F,numInst,i);
        U(:,i) = (First_item(:,i)+ Sec_item(:,i) - (1/2)*lambda3*all')./(F_item(:,i) + S_item(:,i));
    end
    for num = 1:numInst
        indnum = [1:numInst];
        indnum(num) = [];
        U(indnum',num) = (EProjSimplex_new(U(indnum',num)'))';
    end
    clear First_item Sec_item F_item S_item all
    linshi_U = (U+U')/2;
    LU = diag(sum(linshi_U))-linshi_U;
    [F, ~, ev]=eig1(LU, num_cluster, 0);
    vec_D = [];
    for iv = 1:length(Sor)
        vec_D = [vec_D,(Sor{iv}(:))];
    end
    for iv = 1:length(Sor)
        W = ones(numInst,numInst);
        ind_0 = find(ind_folds(:,iv) == 0);  
        W(:,ind_0) = 0;
        W(ind_0,:) = 0;
        Rec_error(iv) = norm(((1/2)*(U+D{iv}).*W-Sor{iv}),'fro')^2+lambda1*norm(vec_D(:,iv)-vec_D*B(:,iv))^2+lambda2*Q(iv)*norm(U-A{iv},'fro')^2;
    end
    clear W ind_0 linshi_S LSv
    linshi_U = (U+U')/2;
    LU = diag(sum(linshi_U))-linshi_U;
    obj(iter) = alpha_r*Rec_error' + lambda3*trace(F'*LU*F);
    clear vec_D
    if iter > 2 && abs(obj(iter)-obj(iter-1))<1e-6
        iter;
        break;
    end
    if iter == max_iter && abs(obj(iter)-obj(iter-1))>1e-6
        disp('迭代数太大');
    end
end
end
function [all]=distance(F,n,ij);
for ji=1:n
    all(ji)=(norm(F(ij,:)-F(ji,:)))^2;
end
end
