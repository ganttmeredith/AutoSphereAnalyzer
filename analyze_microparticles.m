% Microparticle analysis using MATLAB regionprops.
% Run in MATLAB with Image Processing Toolbox:
%   analyze_microparticles

imgPath = fullfile(fileparts(mfilename('fullpath')), 'input_image.png');
outDir = fullfile(fileparts(mfilename('fullpath')), 'output');
particleDir = fullfile(outDir, 'particles');

if ~exist(outDir, 'dir'); mkdir(outDir); end
if ~exist(particleDir, 'dir'); mkdir(particleDir); end

I = imread(imgPath);
if size(I, 3) >= 3
    gray = rgb2gray(I(:, :, 1:3));
else
    gray = I;
end
gray = im2double(gray);

[h, w] = size(gray);
barStrip = gray(round(h * 0.90):end, :);
dark = barStrip < 0.35;
umPerPx = 100 / 90; % fallback
for y = 1:size(dark, 1)
    row = dark(y, :);
    d = diff([0, row, 0]);
    starts = find(d == 1);
    ends = find(d == -1) - 1;
    for k = 1:numel(starts)
        len = ends(k) - starts(k) + 1;
        if len >= 40 && len <= 250
            umPerPx = 100 / len;
        end
    end
end

cropH = round(h * 0.88);
roi = gray(1:cropH, :);
tophat = im2double(imtophat(im2uint8(roi), strel('disk', 25)));
level = graythresh(tophat(tophat > 0.01));
bw = tophat > level * 0.45;
bw = bwareaopen(bw, 80);
bw = imclose(bw, strel('disk', 3));
bw = imopen(bw, strel('disk', 2));
bw = imfill(bw, 'holes');

L = bwlabel(bw);
S = regionprops(L, roi, ...
    'Area', 'Perimeter', 'BoundingBox', 'Centroid', ...
    'MajorAxisLength', 'MinorAxisLength', 'Orientation', ...
    'Eccentricity', 'Solidity', 'Extent', 'EquivDiameter', ...
    'EulerNumber', 'MeanIntensity', 'MaxIntensity', 'MinIntensity');

% Remove large dark crater-like regions.
keep = true(numel(S), 1);
for i = 1:numel(S)
    areaUm2 = S(i).Area * umPerPx^2;
    if areaUm2 > 6500 && S(i).MaxIntensity < 0.48
        keep(i) = false;
    elseif S(i).Area < 350
        keep(i) = false;
    end
end
S = S(keep);
[~, ord] = sort([S.Area], 'descend');
S = S(ord);

T = struct2table(S);
T.particle_id = (1:height(T))';
T.major_axis_length_um = T.MajorAxisLength * umPerPx;
T.minor_axis_length_um = T.MinorAxisLength * umPerPx;
T.area_um2 = T.Area * umPerPx^2;
T.equiv_diameter_um = T.EquivDiameter * umPerPx;

for i = 1:numel(S)
    bb = round(S(i).BoundingBox); % [x y w h]
    pad = 18;
    r0 = max(1, bb(2) - pad);
    c0 = max(1, bb(1) - pad);
    r1 = min(cropH, bb(2) + bb(4) + pad);
    c1 = min(size(roi, 2), bb(1) + bb(3) + pad);
    crop = roi(r0:r1, c0:c1);

    fig = figure('Visible', 'off', 'Color', 'w');
    imshow(crop, []);
    hold on;
    rectangle('Position', [bb(1)-c0+1, bb(2)-r0+1, bb(3), bb(4)], ...
        'EdgeColor', [0 1 0], 'LineWidth', 1.8);
    cy = S(i).Centroid(1) - r0 + 1;
    cx = S(i).Centroid(2) - c0 + 1;
    theta = deg2rad(S(i).Orientation);
    maj = S(i).MajorAxisLength / 2;
    minr = S(i).MinorAxisLength / 2;
    plot(cx + [-1 1] * maj * cos(theta), cy + [-1 1] * maj * sin(theta), 'r-', 'LineWidth', 2);
    plot(cx + [-1 1] * minr * cos(theta + pi/2), cy + [-1 1] * minr * sin(theta + pi/2), 'c-', 'LineWidth', 2);
    title(sprintf('Particle %d', i));
    outPng = fullfile(particleDir, sprintf('particle_%03d.png', i));
    exportgraphics(fig, outPng, 'Resolution', 150);
    close(fig);
end

writetable(T, fullfile(outDir, 'particle_measurements_matlab.xlsx'), 'Sheet', 'measurements');
fprintf('Saved %d particles to %s\n', numel(S), particleDir);
