function [h,I] = plot_RF (A, dim_patch, dim_topo, figdir, figname, flag_normalize, flag_colormap, method_2D, h)


if (nargin<6)
    flag_normalize = 0;
end

if (nargin<7)
    flag_colormap = 0;
end

if (nargin<8)
    method_2D = 1;
end


%% normalize each patch

if (flag_normalize==1)
    A = A./(ones(size(A,1),1)*max(abs(A)));
elseif (flag_normalize==2)
    A = (A-ones(size(A,1),1)*min(A)) ./ (ones(size(A,1),1)*(max(A)-min(A)));
end

max_abs_A = max(abs(A(:)));


%% prepare

% subplot grid
if (dim_topo(1)==1)
    n_col = ceil(sqrt(dim_topo(2)));
    n_row = ceil(dim_topo(2)/n_col);
else
    n_col = dim_topo(2);
    n_row = dim_topo(1);
end

% create new figure or just keep original
if (nargin<9)
    figure;
    if (~flag_colormap)
        colormap(gray);
    end
    if ((dim_patch(1)~=1)&&(method_2D==1))
        h = 0;
    else
        h = cell(1,prod(dim_topo));
    end
end


%% plot

if (dim_patch(1)~=1)            % 2D patch
    
    if (method_2D)

        % 1 pixel boundary
        dim_patch1 = dim_patch+1;

        % Initialization of the image
        I = ones(dim_patch1(1)*n_row-1,dim_patch1(2)*n_col-1);

        % Transfer features to this image matrix
        for idx = 1:prod(dim_topo)
            A_patch = flipud(reshape(A(:,idx),dim_patch));
           [i_col,i_row] = ind2sub([n_col,n_row],idx); % scan horizontally
           % [i_row,i_col] = ind2sub([n_row,n_col],idx); % scan verically
            roi_y = (i_row-1)*dim_patch1(1)+(1:dim_patch(1));
            roi_x = (i_col-1)*dim_patch1(2)+(1:dim_patch(2));
            I(roi_y , roi_x) = A_patch;
        end

        % plot
        if (h==0)
            h = imagesc(I,[-max_abs_A max_abs_A]); 
            axis equal tight off;
            xlabel('X'); ylabel('Y');
        else
            set(h,'CData',I);
            set(gca,'CLim',[-max_abs_A max_abs_A]);
            xlabel('X'); ylabel('Y');
        end
        drawnow;
        
    else
        
        for idx = 1:prod(dim_topo)
            A_patch = reshape(A(:,idx),dim_patch);

            if (isempty(h{idx}))
                subplot(n_row,n_col, idx);
                h{idx} = imagesc(1:dim_patch(2), 1:dim_patch(1), A_patch);
                set(gca,'CLim',[-max_abs_A max_abs_A]);
                axis equal tight off;
                xlabel('X'); ylabel('Y');
            else
                set(h{idx},'CData',A_patch);
                set(gca,'CLim',[-max_abs_A max_abs_A]);
                xlabel('X'); ylabel('Y');
            end
        end
        drawnow;
        
    end
    
else                            % 1D patch
   
    % get plot rows and cols if not provided
    if (length(dim_patch)==2)   
        % 1D patch, 1D topo, one type (i.e. left/right, real/iamg)
        A = reshape(A,dim_patch(2),1,prod(dim_topo));
    else
        % 1D patch, 1D topo, several type (i.e. left/right or real/imag)
        A = reshape(A,dim_patch(2),dim_patch(3),prod(dim_topo));
    end
    
    % plot
    for idx = 1:prod(dim_topo)
        if (isempty(h{idx}))
            subplot(n_row,n_col, idx);
            h{idx} = plot(1:dim_patch(2), A(:,:,idx));
            
            if (method_2D==1)       % JUST REUSE THE VARIABLE FOR OTHER PURPOSE
                set(gca,'xTick',[],'YTick',[],'XLim',[0 dim_patch(2)],'YLim',[-max_abs_A max_abs_A]);
            else
                set(gca,'xTick',[],'YTick',[],'XLim',[0 dim_patch(2)],'YLim',[0 max_abs_A]);
            end
            xlabel('X'); ylabel('Y');
%             saveas(gcf,[figdir,figname,'/',num2str(idx),'.fig']); close;
        else
            if (length(dim_patch)==2) 
                set(h{idx},'YData',A(:,1,idx));
            else
                for idx_type = 1:dim_patch(3)
                    set(h{idx}(idx_type),'YData',A(:,idx_type,idx));
                end
            end
            set(gca,'YLim',[-max_abs_A max_abs_A]);
        end
    end
    drawnow;
    
end


%% save figure

if ((nargin>3)&&(~isempty(figdir))&&(~isempty(figname)))
    saveas(gcf,[figdir,figname,'.fig']); close;
end

