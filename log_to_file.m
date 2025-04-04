function log_to_file(Synthesis, filename, varargin)

    if nargin == 2
        optiontable = 0;
    else
        optiontable = 1;
    end

    % SAVE_RESULTS Displays and logs the synthesis table
    % 
    % This function prints the Synthesis table to the console using disp()
    % and saves the results to the  text file filename.
    %
    % Inputs:
    %   - Synthesis: A table or matrix containing the results
    %   - filename: Name of the text file where the results will be saved
    %
    % Example usage:
    %   save_results(Synthesis, 'Results/synthesis_report.txt');

    % Ensure the directory exists
    results_dir = fileparts(filename); % Extract directory path
    if ~isempty(results_dir) && ~exist(results_dir, 'dir')
        mkdir(results_dir);
    end

    % Display results in the command window
    %disp('Synthesis Results:');
    disp(Synthesis);

    % Open file for appending
    fid = fopen(filename, 'a');
    if fid == -1
        error('Error opening the file: %s', filename);
    end

    % Write results to the file
    %fprintf(fid, 'Synthesis Results:\n\n');
    
    % If Synthesis is a table, convert it to text and write it
    if istable(Synthesis)
        if optiontable == 0
            % Extract variable names and values
            var_names = Synthesis.Properties.VariableNames; % Column names
            values = table2array(Synthesis); % Convert table to numerical array

            % Write results line by line (each variable name with its value)
            for i = 1:numel(var_names)
                fprintf(fid, '%s\t%.6f\n', var_names{i}, values(1, i)); 
            end
        else
           writetable(Synthesis, filename, 'WriteMode', 'append', 'WriteVariableNames', true, 'Delimiter', 'tab');
        end
    else
        % If it's a matrix, write manually
        for i = 1:size(Synthesis, 1)
            fprintf(fid, '%s\n', num2str(Synthesis(i, :)));
        end
    end
    %fprintf(fid, '\n'); % Adds a blank line
    
    % Close the file
    fclose(fid);

%    fprintf('Results saved to: %s\n', filename);
end
