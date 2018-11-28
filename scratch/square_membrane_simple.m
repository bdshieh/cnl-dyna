% COMSOL CODE FOR K MATRIX CALCULATION IN MATLAB
% CHECK SELECTION NUMBERS BY SAVING/OPENING MODEL IN COMSOL (mphsave)!

%% Membrane dimensions, material properties, and BEM mesh nodal spacing

elasticX = 45e-6; 
elasticY = 45e-6; 
elasticH = 2.2e-6; 
piezoX = 45e-6;
piezoY = 45e-6;
piezoH = 1e-6;

elasticDensity = 2200;
elasticModulus = 70e9;
elasticPRatio = 0.17;
piezoDensity = 7500;
piezoModulus = 64e9;
piezoPRatio = 0.31;

meshSize = 2; % mesh size for elastic and piezo layers
nNodesX = 5; % number of nodes in X direction (includes clamped nodes)
nNodesY = 5; % number of nodes in Y direction (includes clamped nodes)
saveFilename = 'kmat_square_simple_5x5.mat';

%% Calculate spacing and create interpolation grid

dx = elasticX/(nNodesX - 1);
dy = elasticY/(nNodesY - 1);

% sets up the interpolation grid
xv = (dx:dx:(dx*(nNodesX - 2))) - elasticX/2;
yv = (dy:dy:(dy*(nNodesY - 2))) - elasticY/2;
zv = [elasticH];
[x, y, z] = ndgrid(xv, yv, zv);
nodes = [x(:), y(:), z(:)]';

%% Start COMSOL model

import com.comsol.model.*
import com.comsol.model.util.*
model = ModelUtil.create('Model');
model.modelNode.create('mod1');
model.geom.create('geom1', 3);
% model.mesh.create('mesh1', 'geom1');
model.physics.create('solid', 'SolidMechanics', 'geom1');
model.study.create('std1');
model.study('std1').feature.create('stat', 'Stationary');

%% Create initial geometry

blk1 = model.geom('geom1').feature.create('blk1', 'Block');
blk1.setIndex('size', num2str(elasticX), 0);
blk1.setIndex('size', num2str(elasticY), 1);
blk1.setIndex('size', num2str(elasticH), 2);
blk1.setIndex('pos', num2str(-elasticX/2), 0);
blk1.setIndex('pos', num2str(-elasticY/2), 1);
model.geom('geom1').run('blk1');

blk2 = model.geom('geom1').feature.create('blk2', 'Block');
blk2.setIndex('size', num2str(dx), 0);
blk2.setIndex('size', num2str(dy), 1);
blk2.setIndex('size', num2str(elasticH/2), 2);
blk2.setIndex('pos', num2str(dx - elasticX/2), 0);
blk2.setIndex('pos', num2str(dy - elasticY/2), 1);
blk2.setIndex('pos', num2str(elasticH/4), 2);
blk2.set('base', 'center');
model.geom('geom1').run('blk2');

blk4 = model.geom('geom1').feature.create('blk4', 'Block');
blk4.setIndex('size', num2str(piezoX), 0);
blk4.setIndex('size', num2str(piezoY), 1);
blk4.setIndex('size', num2str(piezoH), 2);
blk4.setIndex('pos', num2str(-piezoX/2), 0);
blk4.setIndex('pos', num2str(-piezoY/2), 1);
blk4.setIndex('pos', num2str(elasticH), 2);
model.geom('geom1').run('blk4');

model.geom('geom1').run;
model.geom('geom1').runAll;
model.view('view1').set('renderwireframe', true);

%% Define domain selections

% elastic domain
x = elasticX;
y = elasticY;
z0 = 0;
z1 = elasticH;
elastic = mphselectbox(model, 'geom1', [-x/2 -y/2 z0; x/2 y/2 z1]', 'domain');

% piezo domain
x = piezoX;
y = piezoY;
z0 = elasticH;
z1 = elasticH + piezoH;
piezo = mphselectbox(model, 'geom1', [-x/2 -y/2 z0; x/2 y/2 z1]', 'domain');

if any(cellfun(@isempty, {elastic piezo}))
    error('Error: A domain selection is empty.');
end

%% Define boundary selections

% elastic fixed boundaries
x = elasticX/2;
y = elasticY/2;
z0 = 0;
z1 = elasticH;
el1 = mphselectcoords(model, 'geom1', [-x -y z0; x -y z1]', 'boundary');
el2 = mphselectcoords(model, 'geom1', [x -y z0; x y z1]', 'boundary');
el3 = mphselectcoords(model, 'geom1', [x y z0; -x y z1]', 'boundary');
el4 = mphselectcoords(model, 'geom1', [-x y z0; -x -y z1]', 'boundary');

% piezo fixed boundaries
x = piezoX/2;
y = piezoY/2;
z0 = elasticH;
z1 = elasticH + piezoH;
pi1 = mphselectcoords(model, 'geom1', [-x -y z0; x -y z1]', 'boundary');
pi2 = mphselectcoords(model, 'geom1', [x -y z0; x y z1]', 'boundary');
pi3 = mphselectcoords(model, 'geom1', [x y z0; -x y z1]', 'boundary');
pi4 = mphselectcoords(model, 'geom1', [-x y z0; -x -y z1]', 'boundary');

% applied load boundary
x0 = dx - elasticX/2 - dx/2;
x1 = dx - elasticX/2 + dx/2;
y0 = dy - elasticY/2 - dy/2;
y1 = dy - elasticY/2 + dy/2;
z = elasticH/2;
al1 = mphselectcoords(model, 'geom1', [x0 y0 z; x1 y1 z]', 'boundary');

% mphviewselection(model, 'geom1', el4, 'boundary')

if any(cellfun(@isempty, {el1 el2 el3 el4 pi1 pi2 pi3 pi4 al1}))
    error('Error: A boundary selection is empty.');
end

%% Set material properties

mat1 = model.material.create('mat1');
mat1.propertyGroup('def').set('youngsmodulus', {num2str(elasticModulus)});
mat1.propertyGroup('def').set('poissonsratio', {num2str(elasticPRatio)});
mat1.propertyGroup('def').set('density', {num2str(elasticDensity)});
mat1.selection.set([elastic]);

mat2 = model.material.create('mat2');
mat2.selection.set([piezo]);
mat2.propertyGroup('def').set('youngsmodulus', {num2str(piezoModulus)});
mat2.propertyGroup('def').set('poissonsratio', {num2str(piezoPRatio)});
mat2.propertyGroup('def').set('density', {num2str(piezoDensity)});

%% Set boundary conditions

fixed = [el1 el2 el3 el4 pi1 pi2 pi3 pi4];
load = [al1];

model.physics('solid').feature.create('fix1', 'Fixed', 2);
model.physics('solid').feature('fix1').selection.set(fixed);

model.physics('solid').feature.create('bndl1', 'BoundaryLoad', 2);
model.physics('solid').feature('bndl1').selection.set(load);
model.physics('solid').feature('bndl1').set('FperArea', {'0' '0' '1'});

%% Mesh geometry

mesh1 = model.mesh.create('mesh1', 'geom1');

ftet1 = mesh1.feature.create('ftet1', 'FreeTet');
ftet1.selection.set([elastic piezo]);
ftet1.feature.create('size1', 'Size');
ftet1.feature('size1').set('hauto', num2str(meshSize));

mesh1.run;

%% Set study

model.sol.create('sol1');
model.sol('sol1').study('std1');
model.sol('sol1').feature.create('st1', 'StudyStep');
model.sol('sol1').feature('st1').set('study', 'std1');
model.sol('sol1').feature('st1').set('studystep', 'stat');
model.sol('sol1').feature.create('v1', 'Variables');
model.sol('sol1').feature.create('s1', 'Stationary');
model.sol('sol1').feature('s1').feature.create('fc1', 'FullyCoupled');
model.sol('sol1').feature('s1').feature.remove('fcDef');
model.sol('sol1').attach('std1');

model.result.create('pg1', 3);
model.result('pg1').set('data', 'dset1');
model.result('pg1').feature.create('surf1', 'Surface');
model.result('pg1').feature('surf1').set('expr', {'solid.mises'});
model.result('pg1').name('Stress (solid)');
model.result('pg1').feature('surf1').feature.create('def', 'Deform');
model.result('pg1').feature('surf1').feature('def').set('expr', {'u' 'v' 'w'});
model.result('pg1').feature('surf1').feature('def').set('descr', 'Displacement field (Material)');

%% CHECK SELECTION NUMBERS HERE !
% mphsave(model, 'temp.mph');

%% set the loop to iterate over all BEM nodes

nnodes = size(nodes, 2);
x = zeros(nnodes, nnodes);

for i = 1:nnodes
        
    disp([num2str(i) '/' num2str(nnodes)])

    locx = nodes(1, i);
    locy = nodes(2, i);
    blk2.setIndex('pos', num2str(locx), 0);
    blk2.setIndex('pos', num2str(locy), 1);
    model.geom('geom1').run;
    model.geom('geom1').runAll;

    mphgeom(model, 'geom1', 'facealpha', '0.1')
    drawnow;

    model.mesh('mesh1').run;

    model.sol('sol1').study('std1');
    model.sol('sol1').feature.remove('s1');
    model.sol('sol1').feature.remove('v1');
    model.sol('sol1').feature.remove('st1');
    model.sol('sol1').feature.create('st1', 'StudyStep');
    model.sol('sol1').feature('st1').set('study', 'std1');
    model.sol('sol1').feature('st1').set('studystep', 'stat');
    model.sol('sol1').feature.create('v1', 'Variables');
    model.sol('sol1').feature.create('s1', 'Stationary');
    model.sol('sol1').feature('s1').feature.create('fc1', 'FullyCoupled');
    model.sol('sol1').feature('s1').feature.remove('fcDef');
    model.sol('sol1').attach('std1');
    model.sol('sol1').runAll;
    model.result('pg1').run;

    data = mpheval(model, 'w');
    mem_disp = mphinterp(model, 'w', 'coord', nodes);
    x(i,:) = mem_disp;

end

save(saveFilename, 'x');