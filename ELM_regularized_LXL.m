function [TrainingTime, TestingTime, test_sensitivity, test_specificity, test_gmean] = ELM_regularized_LXL(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction,C)


%%%%%%%%%%% Macro definition
REGRESSION=0;
CLASSIFIER=1;

%%%%%%%%%%% Load training dataset
train_data = TrainingData_File;
T=train_data(:,1)';
P=train_data(:,2:size(train_data,2))';
clear train_data;                                   %   Release raw training data array

%%%%%%%%%%% Load testing dataset
test_data =  TestingData_File;
TV.T=test_data(:,1)';
TV.P=test_data(:,2:size(test_data,2))';
clear test_data;                                    %   Release raw testing data array


NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);

[x,posx] = find(T==1);
W = zeros(length(T),length(T));

for i = 1:length(T)
    if T(i) == 1
        W(i,i)= 0.618/length(posx);
    else
        W(i,i)= 1/(length(T)-length(posx));
    end 
end

%save W W;

if Elm_Type~=REGRESSION
    %%%%%%%%%%%% Preprocessing the data of classification
    sorted_target=sort(cat(2,T,TV.T),2);
    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(NumberofTrainingData+NumberofTestingData)
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;
    NumberofOutputNeurons=number_class;
    
    %%%%%%%%%% Processing the targets of training
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                break; 
            end
        end
        temp_T(j,i)=1;
    end
    T=temp_T*2-1;

    %%%%%%%%%% Processing the targets of testing
    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(1,j) == TV.T(1,i)
                break; 
            end
        end
        temp_TV_T(j,i)=1;
    end
    TV.T=temp_TV_T*2-1;

end                                                 %   end if of Elm_Type



%%%%%%%%%%% Calculate weights & biases
% start_time_train=cputime;
tic;
%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
tempH=InputWeight*P;
clear P;                                            %   Release input of training data 
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;

%%%%%%%%%%% Calculate hidden neuron output matrix H
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = double(hardlim(tempH));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(tempH);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H = radbas(tempH);
        %%%%%%%% More activation functions can be added here                
end
clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H

%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
% OutputWeight=pinv(H') * T';                        % slower implementation
% n = size(T,2);
% OutputWeight=H'*((H'*H+speye(n)/C)\(T')); 
n = NumberofHiddenNeurons;

OutputWeight=((H*W*H'+speye(n)/C)\(H*W*T')); 
% OutputWeight=mtimesx(H,((mtimesx(H',H)+speye(n)/C)\T')); 
% OutputWeight=inv(H * H') * H * T';                         % faster implementation
% end_time_train=cputime;
% TrainingTime=end_time_train-start_time_train     ;   %   Calculate CPU time (seconds) spent for training ELM
TrainingTime=toc;
%%%%%%%%%%% Calculate the training accuracy
Y=(H' * OutputWeight)';                             %   Y: the actual output of the training data
if Elm_Type == REGRESSION
    TrainingAccuracy=sqrt(mse(T - Y))  ;             %   Calculate training accuracy (RMSE) for regression case
end
clear H;

%%%%%%%%%%% Calculate the output of testing input
start_time_test=cputime;
tempH_test=InputWeight*TV.P;
clear TV.P;             %   Release input of testing data             
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);        
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = hardlim(tempH_test);        
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H_test = tribas(tempH_test);        
    case {'radbas'}
        %%%%%%%% Radial basis function
        H_test = radbas(tempH_test);        
        %%%%%%%% More activation functions can be added here        
end
TY=(H_test' * OutputWeight)';                       %   TY: the actual output of the testing data
end_time_test=cputime;
TestingTime=end_time_test-start_time_test      ;     %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data

if Elm_Type == REGRESSION
    TestingAccuracy=sqrt(mse(TV.T - TY))       ;     %   Calculate testing accuracy (RMSE) for regression case
end

if Elm_Type == CLASSIFIER
%%%%%%%%%% Calculate training & testing classification accuracy
    %MissClassificationRate_Training=0;
    %MissClassificationRate_Testing=0;

    TP=0;
    TN=0;
    FP=0;
    FN=0;
    
    for i = 1 : size(T, 2)
        [x, label_index_expected]=max(T(:,i));
        [x, label_index_actual]=max(Y(:,i));
        if label_index_expected == 1 & label_index_actual ==1
            TP = TP + 1;
        elseif label_index_expected == 1 & label_index_actual ==2
            FN = FN + 1;
        elseif label_index_expected == 2 & label_index_actual ==1
            FP = FP + 1;
        elseif label_index_expected == 2 & label_index_actual ==2
            TN = TN + 1;
        end
        %if label_index_actual~=label_index_expected
        %    MissClassificationRate_Training=MissClassificationRate_Training+1;
        %end
    end
    %TrainingAccuracy=1-MissClassificationRate_Training/size(T,2)  ;
    train_sensitivity = TP/(TP+FN);
    train_specificity = TN/(TN+FP);
    train_gmean = sqrt(train_sensitivity * train_specificity);
    
    TP=0;
    TN=0;
    FP=0;
    FN=0;
    for i = 1 : size(TV.T, 2)
        [x, label_index_expected]=max(TV.T(:,i));
        [x, label_index_actual]=max(TY(:,i));
        %if label_index_actual~=label_index_expected
        %%    MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        %end
        if label_index_expected == 1 & label_index_actual ==1
            TP = TP + 1;
        elseif label_index_expected == 1 & label_index_actual ==2
            FN = FN + 1;
        elseif label_index_expected == 2 & label_index_actual ==1
            FP = FP + 1;
        elseif label_index_expected == 2 & label_index_actual ==2
            TN = TN + 1;
        end
    end
    %TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2)  ;
    test_sensitivity = TP/(TP+FN);
    test_specificity = TN/(TN+FP);
    test_gmean = sqrt(test_sensitivity * test_specificity);
end