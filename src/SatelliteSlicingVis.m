classdef SatelliteSlicingVis < matlab.apps.AppBase

    properties (Access = public)
        UIFigure             matlab.ui.Figure
        MainLayout           matlab.ui.container.GridLayout
        
        % Heading
        HeadingLabel         matlab.ui.control.Label

        % Panels (Containers)
        LeftPanel            matlab.ui.container.Panel
        RightPanel           matlab.ui.container.Panel
        KeyPanel             matlab.ui.container.Panel
        DataPanel            matlab.ui.container.Panel
        ControlsPanel        matlab.ui.container.Panel

        % Inner Layouts (To make things stretch)
        LeftLayout           matlab.ui.container.GridLayout
        RightLayout          matlab.ui.container.GridLayout
        KeyLayout            matlab.ui.container.GridLayout
        DataLayout           matlab.ui.container.GridLayout
        ControlsLayout       matlab.ui.container.GridLayout

        % Components
        URLLCAxes            matlab.ui.control.UIAxes
        URLLCErrorLabel      matlab.ui.control.Label
        EMBBAxes             matlab.ui.control.UIAxes
        EMBBErrorLabel       matlab.ui.control.Label
        KeyAxes              matlab.ui.control.UIAxes
        DataAxes             matlab.ui.control.UIAxes
        
        % Controls
        OrbitSlider          matlab.ui.control.Slider
        OrbitSliderLabel     matlab.ui.control.Label
        SNRLabel             matlab.ui.control.Label
        AngleLabel           matlab.ui.control.Label
    end

    properties (Access = private)
        SatAltitude = 500;      % km
        Frequency = 3.5e9;      % 3.5 GHz
        TxPower = 40;           % dBm
        AntennaGain = 35;       % dB
        NoiseFloor = -100;      % dBm
        K_factor = 10;          
        
        RefQPSK
        Ref64QAM
    end

    methods (Access = private)
        
        function initConstants(app)
            app.RefQPSK = [1+1i, 1-1i, -1+1i, -1-1i].' / sqrt(2);
            [X, Y] = meshgrid(-7:2:7, -7:2:7);
            c = X + 1i*Y;
            app.Ref64QAM = c(:) / sqrt(mean(abs(c(:)).^2));
        end

        function updateSimulation(app)
            % --- PHYSICS & LINK BUDGET ---
            angle_deg = app.OrbitSlider.Value;
            app.AngleLabel.Text = sprintf('Angle: %.1f°', angle_deg);
            
            angle_rad = deg2rad(angle_deg);
            slant_dist_km = app.SatAltitude / cos(angle_rad); 
            if abs(angle_deg) > 80; slant_dist_km = 3000; end 
            
            d_meters = slant_dist_km * 1000;
            path_loss = 20*log10(d_meters) + 20*log10(app.Frequency) - 147.55;
            
            % Steeper roll-off for new range (-45 to 45)
            pattern_loss = 0.1 * abs(angle_deg); 
            
            avg_snr_db = app.TxPower + app.AntennaGain - path_loss - pattern_loss - app.NoiseFloor;
            
            % --- CHANNEL ---
            snr_linear = 10^(avg_snr_db/10);
            noise_power = 1 / snr_linear;
            
            h_los = sqrt(app.K_factor / (app.K_factor + 1));
            h_scat = sqrt(1 / (app.K_factor + 1)) * (randn()+1i*randn())/sqrt(2);
            h = h_los + h_scat; 
            
            inst_snr_db = 10*log10(snr_linear * abs(h)^2);
            app.SNRLabel.Text = sprintf('SNR: %.2f dB', inst_snr_db);

            % --- URLLC (QPSK) ---
            cla(app.URLLCAxes); hold(app.URLLCAxes, 'on');
            urllc_active = inst_snr_db > 3; 
            
            if urllc_active
                nSym = 200;
                tx_sym = app.RefQPSK(randi([1, 4], nSym, 1));
                noise = sqrt(noise_power/2) * (randn(size(tx_sym)) + 1i*randn(size(tx_sym)));
                rx_eq = (h * tx_sym + noise) / h;
                
                plot(app.URLLCAxes, rx_eq, '.', 'Color', [0 0.4470 0.7410], 'MarkerSize', 12);
                
                tx_signs = sign(real(tx_sym)) + 1i*sign(imag(tx_sym));
                rx_signs = sign(real(rx_eq)) + 1i*sign(imag(rx_eq));
                err_rate = (sum(tx_signs ~= rx_signs) / nSym) * 100;
                
                app.URLLCErrorLabel.Text = sprintf("Error: %.2f%%", err_rate);
                if err_rate > 0.1
                     app.URLLCErrorLabel.FontColor = [0.8 0.5 0]; 
                     key_color = [0.8 0.8 0]; key_status = 'WARNING';
                else
                     app.URLLCErrorLabel.FontColor = [0 0.6 0]; 
                     key_color = [0 0.8 0.6]; key_status = 'SECURE';
                end
                
                map = zeros(10,10,3);
                map(:,:,1)=key_color(1); map(:,:,2)=key_color(2); map(:,:,3)=key_color(3);
                imagesc(app.KeyAxes, map);
                title(app.KeyAxes, ['Key State: ' key_status], 'FontSize', 14, 'Color', 'white');
            else
                text(app.URLLCAxes, 0, 0, 'LINK LOST', 'HorizontalAlignment', 'center', 'Color', 'r', 'FontSize', 20, 'FontWeight', 'bold');
                app.URLLCErrorLabel.Text = "Error: 100%";
                app.URLLCErrorLabel.FontColor = 'red';
                
                map = zeros(10,10,3); map(:,:,1)=1; 
                imagesc(app.KeyAxes, map);
                title(app.KeyAxes, 'Key State: BROKEN', 'FontSize', 14, 'Color', 'white');
            end
            
            plot(app.URLLCAxes, real(app.RefQPSK), imag(app.RefQPSK), 'r+', 'MarkerSize', 14, 'LineWidth', 2);
            grid(app.URLLCAxes, 'on'); xlim(app.URLLCAxes,[-2 2]); ylim(app.URLLCAxes,[-2 2]);
            
            % --- eMBB (64-QAM) ---
            cla(app.EMBBAxes); hold(app.EMBBAxes, 'on');
            
            nSym = 800;
            tx_64 = app.Ref64QAM(randi([1, 64], nSym, 1));
            noise_64 = sqrt(noise_power/2) * (randn(size(tx_64)) + 1i*randn(size(tx_64)));
            rx_eq = (h * tx_64 + noise_64) / h;
            
            if inst_snr_db > 22
                embb_err = 0; col = [0 0.6 0]; data_status = 'INTEGRITY OK';
            elseif inst_snr_db > 15
                embb_err = (22 - inst_snr_db)*2; col = [0.8 0.5 0]; data_status = 'RETRANSMITTING';
            else
                embb_err = 100; col = 'red'; data_status = 'CORRUPTED';
            end
            
            plot(app.EMBBAxes, rx_eq, '.', 'Color', [0.85 0.33 0.1], 'MarkerSize', 6);
            plot(app.EMBBAxes, real(app.Ref64QAM), imag(app.Ref64QAM), 'w.', 'MarkerSize', 2);
            
            app.EMBBErrorLabel.Text = sprintf("Error: %.2f%%", embb_err);
            app.EMBBErrorLabel.FontColor = col;
            
            grid(app.EMBBAxes, 'on'); xlim(app.EMBBAxes,[-2 2]); ylim(app.EMBBAxes,[-2 2]);
            
            data_grid = rand(15, 30);
            if inst_snr_db < 18
                noise_lvl = (18 - inst_snr_db)/10; 
                if noise_lvl > 1; noise_lvl = 1; end
                mask = rand(15,30) < noise_lvl;
                data_grid(mask) = 0; data_grid = data_grid + 0.5*mask; 
            end
            imagesc(app.DataAxes, data_grid); colormap(app.DataAxes, 'hot');
            title(app.DataAxes, ['Data State: ' data_status], 'FontSize', 14, 'Color', 'white');
            app.DataAxes.XTick=[]; app.DataAxes.YTick=[];
        end
    end

    methods (Access = public)
        function app = SatelliteSlicingVis
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [50 50 1200 850]; 
            app.UIFigure.Name = '5G Satellite Slicing Demo';
            app.UIFigure.Color = [0.1 0.1 0.12]; % Dark Theme Background

            initConstants(app);

            % MAIN LAYOUT
            app.MainLayout = uigridlayout(app.UIFigure);
            app.MainLayout.ColumnWidth = {'1x', '1x'};
            % Row weights: Heading (50), Plots (1x), States (1x), Controls (80)
            app.MainLayout.RowHeight = {50, '1x', '1x', 80}; 
            app.MainLayout.RowSpacing = 5;
            app.MainLayout.ColumnSpacing = 5;
            app.MainLayout.Padding = [5 5 5 5];

            % --- 1. HEADER ---
            app.HeadingLabel = uilabel(app.MainLayout);
            app.HeadingLabel.Layout.Row = 1;
            app.HeadingLabel.Layout.Column = [1 2];
            app.HeadingLabel.Text = '5G Network Slicing on Satellite Link';
            app.HeadingLabel.FontSize = 24;
            app.HeadingLabel.FontWeight = 'bold';
            app.HeadingLabel.HorizontalAlignment = 'center';
            app.HeadingLabel.FontColor = [0.4 0.7 1.0]; % Light Blue

            % --- 2. PLOTS (Constellations) ---
            
            % LEFT PANEL (URLLC)
            app.LeftPanel = uipanel(app.MainLayout);
            app.LeftPanel.Layout.Row = 2;
            app.LeftPanel.Layout.Column = 1;
            app.LeftPanel.BackgroundColor = 'black';
            app.LeftPanel.BorderType = 'none';
            
            % Inner Layout for Left Panel (To Stretch Axes)
            app.LeftLayout = uigridlayout(app.LeftPanel);
            app.LeftLayout.ColumnWidth = {'1x'};
            app.LeftLayout.RowHeight = {'1x', 30}; % Plot takes all space, Label takes 30
            app.LeftLayout.Padding = [0 0 0 0];
            
            app.URLLCAxes = uiaxes(app.LeftLayout);
            app.URLLCAxes.Layout.Row = 1;
            app.URLLCAxes.Color = 'black';
            app.URLLCAxes.XColor = 'white'; app.URLLCAxes.YColor = 'white';
            app.URLLCAxes.GridColor = [0.3 0.3 0.3];
            title(app.URLLCAxes, 'URLLC (QPSK)', 'Color', 'white', 'FontSize', 14);

            app.URLLCErrorLabel = uilabel(app.LeftLayout);
            app.URLLCErrorLabel.Layout.Row = 2;
            app.URLLCErrorLabel.Text = 'Error: --';
            app.URLLCErrorLabel.FontSize = 18;
            app.URLLCErrorLabel.FontWeight = 'bold';
            app.URLLCErrorLabel.HorizontalAlignment = 'center';
            app.URLLCErrorLabel.FontColor = 'white';

            % RIGHT PANEL (eMBB)
            app.RightPanel = uipanel(app.MainLayout);
            app.RightPanel.Layout.Row = 2;
            app.RightPanel.Layout.Column = 2;
            app.RightPanel.BackgroundColor = 'black';
            app.RightPanel.BorderType = 'none';
            
            % Inner Layout for Right Panel
            app.RightLayout = uigridlayout(app.RightPanel);
            app.RightLayout.ColumnWidth = {'1x'};
            app.RightLayout.RowHeight = {'1x', 30};
            app.RightLayout.Padding = [0 0 0 0];

            app.EMBBAxes = uiaxes(app.RightLayout);
            app.EMBBAxes.Layout.Row = 1;
            app.EMBBAxes.Color = 'black';
            app.EMBBAxes.XColor = 'white'; app.EMBBAxes.YColor = 'white';
            app.EMBBAxes.GridColor = [0.3 0.3 0.3];
            title(app.EMBBAxes, 'eMBB (64-QAM)', 'Color', 'white', 'FontSize', 14);

            app.EMBBErrorLabel = uilabel(app.RightLayout);
            app.EMBBErrorLabel.Layout.Row = 2;
            app.EMBBErrorLabel.Text = 'Error: --';
            app.EMBBErrorLabel.FontSize = 18;
            app.EMBBErrorLabel.FontWeight = 'bold';
            app.EMBBErrorLabel.HorizontalAlignment = 'center';
            app.EMBBErrorLabel.FontColor = 'white';

            % --- 3. STATES ---
            
            % Key Panel
            app.KeyPanel = uipanel(app.MainLayout);
            app.KeyPanel.Layout.Row = 3;
            app.KeyPanel.Layout.Column = 1;
            app.KeyPanel.BackgroundColor = [0.1 0.1 0.1];
            
            app.KeyLayout = uigridlayout(app.KeyPanel);
            app.KeyLayout.ColumnWidth = {'1x'};
            app.KeyLayout.RowHeight = {'1x'};
            app.KeyLayout.Padding = [5 5 5 5];
            
            app.KeyAxes = uiaxes(app.KeyLayout);
            app.KeyAxes.Color = [0.1 0.1 0.1];
            app.KeyAxes.XColor = 'none'; app.KeyAxes.YColor = 'none';
            if ~isempty(which('disableDefaultInteractivity')); disableDefaultInteractivity(app.KeyAxes); end

            % Data Panel
            app.DataPanel = uipanel(app.MainLayout);
            app.DataPanel.Layout.Row = 3;
            app.DataPanel.Layout.Column = 2;
            app.DataPanel.BackgroundColor = [0.1 0.1 0.1];
            
            app.DataLayout = uigridlayout(app.DataPanel);
            app.DataLayout.ColumnWidth = {'1x'};
            app.DataLayout.RowHeight = {'1x'};
            app.DataLayout.Padding = [5 5 5 5];

            app.DataAxes = uiaxes(app.DataLayout);
            app.DataAxes.Color = [0.1 0.1 0.1];
            app.DataAxes.XColor = 'none'; app.DataAxes.YColor = 'none';
            if ~isempty(which('disableDefaultInteractivity')); disableDefaultInteractivity(app.DataAxes); end

            % --- 4. CONTROLS (Compact) ---
            app.ControlsPanel = uipanel(app.MainLayout);
            app.ControlsPanel.Layout.Row = 4;
            app.ControlsPanel.Layout.Column = [1 2];
            app.ControlsPanel.BackgroundColor = [0.2 0.2 0.25];
            app.ControlsPanel.BorderType = 'none';
            
            % Use Grid Layout for Controls to align perfectly
            app.ControlsLayout = uigridlayout(app.ControlsPanel);
            app.ControlsLayout.ColumnWidth = {150, '1x', 150}; % Label, Slider, SNR
            app.ControlsLayout.RowHeight = {30, 40};
            app.ControlsLayout.Padding = [20 5 20 5];
            
            % Slider Label (Top Left)
            app.OrbitSliderLabel = uilabel(app.ControlsLayout);
            app.OrbitSliderLabel.Layout.Row = 1;
            app.OrbitSliderLabel.Layout.Column = 1;
            app.OrbitSliderLabel.Text = 'Satellite Angle';
            app.OrbitSliderLabel.FontColor = 'white';
            app.OrbitSliderLabel.FontWeight = 'bold';
            app.OrbitSliderLabel.FontSize = 14;

            % Angle Value (Bottom Left)
            app.AngleLabel = uilabel(app.ControlsLayout);
            app.AngleLabel.Layout.Row = 2;
            app.AngleLabel.Layout.Column = 1;
            app.AngleLabel.Text = '0.0°';
            app.AngleLabel.FontColor = [0.4 0.8 1.0];
            app.AngleLabel.FontSize = 18;

            % Slider (Spans 2 Rows in Middle Column)
            app.OrbitSlider = uislider(app.ControlsLayout);
            app.OrbitSlider.Layout.Row = [1 2];
            app.OrbitSlider.Layout.Column = 2;
            app.OrbitSlider.Limits = [-45 45];      % Requested Range
            app.OrbitSlider.MajorTicks = -45:5:45;  % High Resolution Ticks
            app.OrbitSlider.Value = 0;
            app.OrbitSlider.FontColor = 'white';
            app.OrbitSlider.ValueChangedFcn = @(s,e) updateSimulation(app);
            app.OrbitSlider.ValueChangingFcn = @(s,e) updateSimulation(app);

            % SNR Label (Right)
            app.SNRLabel = uilabel(app.ControlsLayout);
            app.SNRLabel.Layout.Row = [1 2];
            app.SNRLabel.Layout.Column = 3;
            app.SNRLabel.Text = 'SNR: -- dB';
            app.SNRLabel.FontColor = 'white';
            app.SNRLabel.FontSize = 16;
            app.SNRLabel.HorizontalAlignment = 'right';
            app.SNRLabel.VerticalAlignment = 'center';

            app.UIFigure.Visible = 'on';
            updateSimulation(app);
        end
    end
end