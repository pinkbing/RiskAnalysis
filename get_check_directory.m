function out = get_check_directory(directoryname)

% Check if the image directory exists; if not, create it
if ~exist(directoryname, 'dir')
    mkdir(directoryname);
    fprintf('Created directory: %s\n', directoryname);
    
end

out = 1;
