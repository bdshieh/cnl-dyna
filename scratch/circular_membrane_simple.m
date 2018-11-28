% COMSOL CODE FOR K MATRIX CALCULATION IN MATLAB
% CHECK SELECTION NUMBERS BY SAVING/OPENING MODEL IN COMSOL (mphsave)!

%% Membrane dimensions, material properties, and BEM mesh nodal spacing

elasticR = 40e-6; 
elasticH = 2.3e-6; 
piezoR = 40e-6;
piezoH = 0.5e-6;

elasticDensity = 2329;
elasticModulus = 170e9;
elasticPRatio = 0.28;
piezoDensity = 3300;
piezoModulus = 348e9;
piezoPRatio = 0.3;

meshSize = 4; % mesh size for elastic and piezo layers
nNodesX = 13;
nNodesY = 13;
saveFilename = 'kmat_akhbari_40um_13x13.mat';

%% Calculate spacing and create interpolation grid

dx = 2*elasticR/(nNodesX - 1);
dy = 2*elasticR/(nNodesY - 1);
edgeBuffer = sqrt((dx/2)^2 + (dy/2)^2);

xv = linspace(-elasticR, elasticR, nNodesX);
yv = linspace(-elasticR, elasticR, nNodesY);
zv = [elasticH/2];

[x, y, z] = ndgrid(xv, yv, zv);
nodes = [x(:), y(:), z(:)]';

% remove nodes not within radius (minus edge buffer)
r = sqrt(nodes(1,:).^2 + nodes(2,:).^2);
nodes(:, r > elasticR - edgeBuffer) = [];

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

blk1 = model.geom('geom1').feature.create('blk1', 'Cylinder');
blk1.set('r', elasticR);
blk1.set('h', elasticH);
blk1.setIndex('pos', num2str(0), 2);
model.geom('geom1').run('blk1');

% square load type
blk2 = model.geom('geom1').feature.create('blk2', 'Block');
blk2.setIndex('size', dx, 0);
blk2.setIndex('size', dy, 1);
blk2.setIndex('size', elasticH/2, 2);
blk2.setIndex('pos', num2str(0), 0);
blk2.setIndex('pos', num2str(0), 1);
blk2.setIndex('pos', elasticH/4, 2);
blk2.set('base', 'center');
model.geom('geom1').run('blk2');

blk3 = model.geom('geom1').feature.create('blk3', 'Cylinder');
blk3.set('r', piezoR);
blk3.set('h', piezoH);
blk3.setIndex('pos', num2str(elasticH), 2);
model.geom('geom1').run('blk3');

model.geom('geom1').run;
model.geom('geom1').runAll;
model.view('view1').set('renderwireframe', true);

%% Define domain selections

% elastic domain
x = elasticR*1.01;
y = elasticR*1.01;
z0 = 0;
z1 = elasticH*1.01;
elastic = mphselectbox(model, 'geom1', [-x -y z0; x y z1]', 'domain');

% piezo domain
x = piezoR;
y = piezoR;
z0 = elasticH;
z1 = elasticH + piezoH;
piezo = mphselectcoords(model, 'geom1', [-x 0 z0; x 0 z1]', 'domain');

if any(cellfun(@isempty, {elastic piezo}))
    error('Error: A domain selection is empty.');
end

%% Define boundary selections

% elastic fixed boundaries
x = elasticR;
y = elasticR;
z0 = 0;
z1 = elasticH;
el1 = mphselectcoords(model, 'geom1', [0 -y z0; x 0 z1]', 'boundary');
el2 = mphselectcoords(model, 'geom1', [x 0 z0; 0 y z1]', 'boundary');
el3 = mphselectcoords(model, 'geom1', [0 y z0; -x 0 z1]', 'boundary');
el4 = mphselectcoords(model, 'geom1', [-x 0 z0; 0 -y z1]', 'boundary');

% piezo fixed boundaries
x = piezoR;
y = piezoR;
z0 = elasticH;
z1 = elasticH + piezoH;
pi1 = mphselectcoords(model, 'geom1', [0 -y z0; x 0 z1]', 'boundary');
pi2 = mphselectcoords(model, 'geom1', [x 0 z0; 0 y z1]', 'boundary');
pi3 = mphselectcoords(model, 'geom1', [0 y z0; -x 0 z1]', 'boundary');
pi4 = mphselectcoords(model, 'geom1', [-x 0 z0; 0 -y z1]', 'boundary');

% applied load boundary
x = dx/2;
y = dy/2;
z = elasticH/2;
al1 = mphselectcoords(model, 'geom1', [-x -y z; x y z]', 'boundary');

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
mphsave(model, 'temp.mph');

%% set the loop to iterate over all BEM nodes

nnodes = size(nodes, 2);
x = zeros(nnodes, nnodes);

for i = 1:nnodes
        
    disp([num2str(i) '/' num2str(nnodes)])

    locx = nodes(1,i);
    locy = nodes(2,i);
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