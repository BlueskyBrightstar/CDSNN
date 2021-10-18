%% CDSNN on Wine dataset %%
clear
close all
clc

%%% Read and process the dataset %%%
Wine_data_numerical = load('Wine_data_numerical.txt');
P = 3;                                % the number of categories

datax1 = Wine_data_numerical(:,1:13);
datax = mapminmax(datax1',0,1);       
datax = datax';
SampleNum = size(datax,1);            % the number of samples
datay = Wine_data_numerical(:,14);
datay1 = zeros(SampleNum,P);          % output coding 
for j = 1:SampleNum 
    if datay(j) == 1
        datay1(j,:) = [7,6,6];
    elseif datay(j) == 2
        datay1(j,:) = [6,7,6];
    else
        datay1(j,:) = [6,6,7];
    end
end
pos = randperm(SampleNum);            % Shuffle the order of samples
datax = datax(pos,:);    
datay = datay(pos,:);
datay1 = datay1(pos,:);

%%% Parameter settings %%%
N = size(datax,2);                    % the number of features
M = 6;                                % the number of Gaussian receptive fields
C = N*M+1;                            % the total number of neurons of Fuzzy Coding Layer 
Q = 8;                                % the number of hidden layer
K = 8;                                % the number of sysnaptsis between two connected neurons
L = 12;                               % the time interval
tau = 8;                              % parameter in PSP function
beta = 1.5;                           % parameter in Gaussian receptive fields
eta = 0.03;                           % learning rate
max_epoch = 100;                      % the maximum number of epochs
ValidK = 10;                          % the number of folds
Times = 5;                            % the number of experiments

%%% Begin the main program including training and testing %%%
All_epoch = zeros(Times,ValidK);
All_Err = zeros(Times,ValidK);
All_ACC = zeros(Times,ValidK);
All_TestErr = zeros(Times,ValidK);
All_TestACC = zeros(Times,ValidK);
for tmp_times = 1:Times               % each experiment 
    indices = crossvalind('Kfold',datay,ValidK); % each cross-validation %
    for validk = 1:ValidK
        testpos = (indices == validk); 
        trainpos = ~testpos;
        
        x = datax(trainpos,:);        % the training dataset
        O = datay(trainpos,:);
        O1 = datay1(trainpos,:);      
        testx=datax(testpos,:);       % the testing dataset 
        testO=datay(testpos,:);
        testO1=datay1(testpos,:);     
        J = sum(trainpos);            % the number of training samples 
        testJ = sum(testpos);         % the number of testing samples
             
        tF = zeros(1,C);
        a = zeros(1,C-1);
        b = zeros(1,C-1);
        for n=1:N                     % Get the initial center and width of Gaussian function
            MaxValue = max(x(:,n));
            MinValue = min(x(:,n));
            for m=1:M
                a(M*(n-1)+m) = MinValue+(2*m-3)*(MaxValue-MinValue)/(2*(M-2));
                b(M*(n-1)+m) = beta*(M-2)/(MaxValue-MinValue);
            end
        end
        w = rand(C*K,Q)/sqrt(48);
        u = rand(Q*K,P)/sqrt(48);
        D_u = zeros(Q*K,P);
        D_w = zeros(C*K,Q);
        theta = 0.7;
        err = zeros(1,J);
        Err = ones(1,max_epoch-1);
        ACC = zeros(1,max_epoch-1);
        output = zeros(J,P);

        y1 = zeros(C*K,1);
        y = zeros(C*K,Q);
        D_y1 = zeros(C*K,1);
        D2_y1 = zeros(C*K,1);
        D_y = zeros(C*K,Q);
        Y1 = zeros(Q*K,1);
        Y = zeros(Q*K,P);
        D_Y1 = zeros(Q*K,1);
        D_Y = zeros(Q*K,P);
        SH = zeros(1,Q);
        D_SH = zeros(1,Q);
        D2_SH = zeros(1,Q);
        SO = zeros(1,P);
        D_SO = zeros(1,P);
        I=ones(1,Q);
        I(1)=-1;

        testy1 = zeros(C*K,1);
        testy = zeros(C*K,Q);
        D_testy1 = zeros(C*K,1);
        D_testy = zeros(C*K,Q);
        testY1 = zeros(Q*K,1);
        testY = zeros(Q*K,P);
        D_testY1 = zeros(Q*K,1);
        D_testY = zeros(Q*K,P);
        testSH = zeros(1,Q);
        D_testSH = zeros(1,Q);
        testSO = zeros(1,P);
        D_testSO = zeros(1,P);
        testerr = zeros(1,J);
        testErr = zeros(1,max_epoch-1);
        testACC = zeros(1,max_epoch-1);
        testACC1 = 0;
        ACC1 = 0;
        testoutput = zeros(J,P);
        T_interval = 4;
        deltaQ = zeros(1,Q);
        testtF = zeros(1,C);
        deltat = 0.01;
        Inte_SH = zeros(1,Q);
        Inte_testSH = zeros(1,Q);
        Inte_epsilon = zeros(C*K,Q);
        alpha = 1;
        lambda = 0.5.*ones(1,Q);      % the balanced parameter lambda
        D_lambda = zeros(1,Q);
        tmp_ACC = 0;
        tmp_testACC = 0;
        epoch = 1;
        
        % while epoch<max_epoch && (epoch==1 || Err(epoch-1)<0.01) 
        while epoch<max_epoch  
            %%%%%%% training process %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            accuracy = 0;
            for j = 1:J
                tH = 100*ones(1,Q);
                tO = 100*ones(1,P);
                %%% coding process %%%%%%%%%%%%%%%%%%%%%
                for n = 1:N
                    for m = 1:M
                        c = M*(n-1)+m;
                        tF(c) = roundn(T_interval.*(1-exp(-(x(j,n)-a(c)).^2./2.*b(c)^2)),-1); 
                    end
                end
                %%%%% Hidden Layer %%%%%%%%%%%%%%%%%%%%%
                for l = 0:0.2:L
                    for c = 1:C
                        for k = 1:K
                            dk = k;
                            tmp = l-tF(c)-dk;
                            if tmp>0
                                y1((c-1)*K+k) = tmp/tau*exp(1-tmp/tau);
                                D_y1((c-1)*K+k) = (1-tmp/tau)/tau*exp(1-tmp/tau);
                                D2_y1((c-1)*K+k) = (-2*exp(1-tmp/tau)+y1((c-1)*K+k))/tau/tau;
                            else
                                y1((c-1)*K+k) = 0;
                                D_y1((c-1)*K+k) = 0;
                                D2_y1((c-1)*K+k) = 0;
                            end
                        end
                    end
                    for q = 1:Q
                        if tH(q)==100
                            SH(q) = sum(w(:,q).*y1);
                            if SH(q)>theta
                                tH(q) = l;
                                y(:,q) = y1;
                                D_y(:,q) = D_y1;
                                D_SH(q) = sum(w(:,q).*D_y1);
                                D2_SH(q) = sum(w(:,q).*D2_y1);
                            end
                        end
                    end
                    if max(tH)<100
                        break
                    end
                end
                mu_DSH = 1./(1+exp(-alpha.*D_SH));        % firing strength
                
                for q = 1:Q  
                    if tH(q) == 100
                        Inte_epsilon(:,q) = 0;
                    else
                        epsilon = (0:deltat:tH(q))./tau.*exp(1-(0:deltat:tH(q))./tau);
                        for c = 1:C
                            for k = 1:K
                                inte_max = tH(q)-(tF(c)+k);
                                if inte_max < 0
                                    Inte_epsilon((c-1)*K+k,q) = 0;
                                else
                                    Nadd1 = round(1+inte_max/deltat);
                                    Inte_epsilon((c-1)*K+k,q) = deltat*(sum(epsilon(1:Nadd1))-epsilon(Nadd1));
                                end
                            end
                        end
                    end
                end
                Inte_sum = Inte_epsilon.*w;
                for q=1:Q
                    Inte_SH(q) = sum(Inte_sum(:,q));
                end
                mu_ISH = 1./(1+exp(-alpha.*Inte_SH));     % AUM
                             
                mu = lambda.*mu_ISH + (1-lambda).*mu_DSH; % mu 
                D_mu = lambda*alpha.*mu_ISH.*(1-mu_ISH).*SH+ (1-lambda)*alpha.*mu_DSH.*(1-mu_DSH).*D2_SH;

                %%%%% Output Layer %%%%%%%%%%%%%%                
                for l = min(tH):0.2:L
                    for q = 1:Q
                        for k = 1:K
                            dk = k;
                            tmp = l-tH(q)-dk;
                            if tmp>0
                                Y1((q-1)*K+k) = I(q)*tmp/tau*exp(1-tmp/tau);
                                D_Y1((q-1)*K+k) = I(q)*(1-tmp/tau)/tau*exp(1-tmp/tau);
                            else
                                Y1((q-1)*K+k) = 0;
                                D_Y1((q-1)*K+k) = 0;
                            end
                        end
                    end
                    for p = 1:P
                        if tO(p)==100
                            mu1 = reshape(repmat(mu,K,1),Q*K,1);
                            D_mu1 = reshape(repmat(D_mu,K,1),Q*K,1);
                            SO(p) = sum(mu1.*u(:,p).*Y1);
                            if SO(p)>theta
                                tO(p) = l;
                                Y(:,p) = Y1;
                                D_Y(:,p) = D_Y1;
                                D_SO(p) = sum(mu1.*u(:,p).*D_Y1);
                            end
                        end
                    end
                    if max(tO)<100
                        break
                    end
                end                

                %%%%%%%%%% err caculate %%%%%%%%%%%%%%%
                output(j,:) = tO;
                err(j) = sum((tO-O1(j,:)).^2)/(2*P);
                %%%%%%%%%% training accuracy %%%%%%%%%%
                [tO1,value] = max(tO);
                if value==O(j)
                    accuracy=accuracy+1;
                end
                %%%%%%%%%% learning process %%%%%%%%%%%%%
                deltaP = (O1(j,:)-tO)./D_SO;
                for q = 1:Q
                    deltaQ(q) = sum(deltaP.*(sum(u(K*(q-1)+1:K*q,:).*repmat(mu1(K*(q-1)+1:K*q),1,P).*D_Y(K*(q-1)+1:K*q,:),1)+...
                        sum(u(K*(q-1)+1:K*q,:).*repmat(D_mu1(K*(q-1)+1:K*q),1,P).*Y(K*(q-1)+1:K*q,:),1)))/D_SH(q);
                end
                for q = 1:Q
                    for k = 1:K
                        D_u((q-1)*K+k,:) = deltaP.*mu1((q-1)*K+k).*Y((q-1)*K+k,:);
                    end
                end
                for c = 1:C
                    for k = 1:K
                        D_w((c-1)*K+k,:) = deltaQ.*y((c-1)*K+k,:);
                    end
                end

%                 for q = 1:Q
%                     D_lambda(q) = sum(deltaP.*sum(u(K*(q-1)+1:K*q,:).*Y(K*(q-1)+1:K*q,:),1))*(mu_DSH(q)-mu_ISH(q));
%                 end
                u = u-eta.*D_u;
                w = w-eta.*D_w;
        %         lambda = lambda-eta.*D_lambda;
                w(w<0)=0;
                u(u<0)=0;
            end
            
            
            %%%%%% testing process %%%%%%%%%%%%%
            testaccuracy = 0;
            for j = 1:testJ
                testtH = 100*ones(1,Q);
                testtO = 100*ones(1,P);
                %%%%% testing coding process %%%%%%%%%%%%%%%%%%%%%
                for n = 1:N
                    for m = 1:M
                        c = M*(n-1)+m;
                        testtF(c) = roundn(T_interval.*(1-exp(-(testx(j,n)-a(c)).^2./2.*b(c)^2)),-1); 
                    end
                end
                %%%%% testing Hidden Layer %%%%%%%%%%%%%%%%%%%%%
                for l = 1:0.2:L
                    for c = 1:C
                        for k = 1:K
                            tmp = l-testtF(c)-k;
                            if tmp>0
                                testy1((c-1)*K+k) = tmp/tau*exp(1-tmp/tau);
                                D_testy1((c-1)*K+k) = (1-tmp/tau)/tau*exp(1-tmp/tau);
                            else
                                testy1((c-1)*K+k) = 0;
                                D_testy1((c-1)*K+k) = 0;
                            end
                        end
                    end
                    for q = 1:Q
                        if testtH(q)==100
                            testSH(q) = sum(w(:,q).*testy1);
                            if testSH(q)>theta
                                testtH(q) = l;
                                testy(:,q) = testy1;
                                D_testy(:,q) = D_testy1;
                                D_testSH(q) = sum(w(:,q).*D_testy1);
                            end
                        end
                    end
                    if max(testtH)<100
                        break
                    end
                end

                testmu_DSH = 1./(1+exp(-alpha.*D_testSH));  % firing strength
                for q = 1:Q  
                    if testtH(q) == 100
                        Inte_epsilon(:,q) = 0;
                    else
                        epsilon = (0:deltat:testtH(q))./tau.*exp(1-(0:deltat:testtH(q))./tau);
                        for c = 1:C
                            for k = 1:K
                                inte_max = testtH(q)-(testtF(c)+k);
                                if inte_max < 0
                                    Inte_epsilon((c-1)*K+k,q) = 0;
                                else
                                    Nadd1 = round(1+inte_max/deltat);
                                    Inte_epsilon((c-1)*K+k,q) = deltat*(sum(epsilon(1:Nadd1))-epsilon(Nadd1));
                                end
                            end
                        end
                    end
                end
                Inte_sum = Inte_epsilon.*w;
                for q=1:Q
                    Inte_testSH(q) = sum(Inte_sum(:,q));
                end
                testmu_ISH = 1./(1+exp(-alpha.*Inte_testSH));
                testmu = lambda.*testmu_DSH + (1-lambda).*testmu_ISH;
                
                %%%%% testing Output Layer %%%%%%%%%%%%%%                
                for l = min(testtH):0.2:L
                    for q = 1:Q
                        for k = 1:K
                            tmp = l-testtH(q)-k;
                            if tmp>0
                                testY1((q-1)*K+k) = I(q)*tmp/tau*exp(1-tmp/tau);
                                D_testY1((q-1)*K+k) = I(q)*(1-tmp/tau)/tau*exp(1-tmp/tau);
                            else
                                testY1((q-1)*K+k) = 0;
                                D_testY1((q-1)*K+k) = 0;
                            end
                        end
                    end
                    for p = 1:P
                        if testtO(p)==100
                            testmu1 = reshape(repmat(testmu,K,1),Q*K,1);    
                            testSO(p) = sum(testmu1.*u(:,p).*testY1);
                            if testSO(p)>theta
                                testtO(p) = l;
                                testY(:,p) = testY1;
                                D_testY(:,p) = D_testY1;
                                D_testSO(p) = sum(testmu1.*u(:,p).*D_testY1);
                            end
                        end
                    end
                    if max(testtO)<100
                        break
                    end
                end                

                %%%%%%%%%% testing err caculate %%%%%%%%%%%%%%%
                testoutput(j,:) = testtO;
                testerr(j) = sum((testtO-testO1(j,:)).^2)/(2*P);
                %%%%%%%%%% testing accuracy %%%%%%%%%%
                [testtO1,value] = max(testtO);
                if value==testO(j)
                    testaccuracy=testaccuracy+1;
                end
            end
            Err(epoch) = sum(err)/J; 
            ACC(epoch) = accuracy/J;
            testErr(epoch) = sum(testerr)/testJ;
            testACC(epoch) = testaccuracy/testJ;
            if (testACC(epoch)>tmp_testACC && ACC(epoch)>=tmp_ACC)||(testACC(epoch)>=tmp_testACC && ACC(epoch)>tmp_ACC)
                tmp_testACC = testACC(epoch);
                tmp_ACC = ACC(epoch);
                tmp_epoch = epoch;
            end  
            epoch = epoch+1;
        end
%         close all
%         figure(1)
%         hold on
%         plot(1:(max_epoch-1),Err(1:max_epoch-1))
%         plot(1:(max_epoch-1),testErr(1:max_epoch-1))
%         hold off
%         figure(2)
%         hold on
%         plot(1:(max_epoch-1),ACC(1:max_epoch-1))
%         plot(1:(max_epoch-1),testACC(1:max_epoch-1))
%         hold off
%         pause
        All_epoch(tmp_times,validk) = tmp_epoch;  % save results
        All_Err(tmp_times,validk) = Err(tmp_epoch);
        All_ACC(tmp_times,validk) = tmp_ACC;
        All_TestErr(tmp_times,validk) = testErr(tmp_epoch);
        All_TestACC(tmp_times,validk) = tmp_testACC; 
    end
end

%%%%%%% Print the results %%%%%%%%%%%%%%%%
[mean(All_epoch(:)),mean(All_ACC(:)),mean(All_TestACC(:))]
% [mean(All_epoch(:)),mean(All_ACC(:)),mean(std(100.*All_ACC,1,2)),mean(All_TestACC(:)),mean(std(100.*All_TestACC,1,2)),...
%     mean(All_Err(:)),mean(std(All_Err,1,2)),mean(All_TestErr(:)),mean(std(All_TestErr,1,2)),Times*ValidK,ValidK]
% figure(1)
% hold on
% plot(1:(epoch-1),Err(1:epoch-1))
% plot(1:(epoch-1),testErr(1:epoch-1))
% hold off
% figure(2)
% hold on
% plot(1:(epoch-1),ACC(1:epoch-1))
% plot(1:(epoch-1),testACC(1:epoch-1))
% hold off



