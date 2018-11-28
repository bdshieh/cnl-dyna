% COMSOL CODE FOR K MATRIX CALCULATION IN MATLAB
% CHECK SELECTION NUMBERS BY SAVING/OPENING MODEL IN COMSOL (mphsave)!

%% Membrane dimensions, material properties, and BEM mesh nodal spacing

% function [x] = comsol_square_membrane(file, verts, lx, ly, lz, rho, ymod, pratio, fine, varea)
function [x] = comsol_square_membrane(verts, lx, ly, lz, rho, ymod, pratio, fine, dx)

    %% Start COMSOL model
    import com.comsol.model.*
    import com.comsol.model.util.*
    model = ModelUtil.create('Model');
    model.modelNode.create('mod1');
    model.geom.create('geom1', 3);
    model.physics.create('solid', 'SolidMechanics', 'geom1');
    model.study.create('std1');
    model.study('std1').feature.create('stat', 'Stationary');

    %% Create initial geometry
    blk1 = model.geom('geom1').feature.create('blk1', 'Block');
    blk1.setIndex('size', num2str(lx), 0);
    blk1.setIndex('size', num2str(ly), 1);
    blk1.setIndex('size', num2str(lz), 2);
    blk1.setIndex('pos', num2str(-lx / 2), 0);
    blk1.setIndex('pos', num2str(-ly / 2), 1);
    model.geom('geom1').run('blk1');

    blk2 = model.geom('geom1').feature.create('blk2', 'Block');
    blk2.setIndex('size', num2str(dx), 0);
    blk2.setIndex('size', num2str(dx), 1);
    blk2.setIndex('size', num2str(lz / 2), 2);
    blk2.setIndex('pos', num2str(dx - lx /2), 0);
    blk2.setIndex('pos', num2str(dx - ly /2), 1);
    blk2.setIndex('pos', num2str(lz / 4), 2);
    blk2.set('base', 'center');
    model.geom('geom1').run('blk2');

    model.geom('geom1').run;
    model.geom('geom1').runAll;
    model.view('view1').set('renderwireframe', true);

    %% Define domain selections
    % elastic domain
    x = lx;
    y = ly;
    z0 = 0;
    z1 = lz;
    elastic = mphselectcoords(model, 'geom1', [-x / 2 -y / 2 z0; x / 2 y / 2 z1]', 'domain');

    if any(cellfun(@isempty, {elastic}))
        error('Error: A domain selection is empty.');
    end

    %% Define boundary selections
    % elastic fixed boundaries
    x = lx / 2;
    y = ly / 2;
    z0 = 0;
    z1 = lz;
    el1 = mphselectcoords(model, 'geom1', [-x -y z0; x -y z1]', 'boundary');
    el2 = mphselectcoords(model, 'geom1', [x -y z0; x y z1]', 'boundary');
    el3 = mphselectcoords(model, 'geom1', [x y z0; -x y z1]', 'boundary');
    el4 = mphselectcoords(model, 'geom1', [-x y z0; -x -y z1]', 'boundary');

    % applied load boundary
    x0 = dx - lx / 2 - dx / 2;
    x1 = dx - lx / 2 + dx / 2;
    y0 = dx - ly / 2 - dx / 2;
    y1 = dx - ly / 2 + dx / 2;
    z = lz / 2;
    al1 = mphselectcoords(model, 'geom1', [x0 y0 z; x1 y1 z]', 'boundary');

    % mphviewselection(model, 'geom1', el4, 'boundary')

    if any(cellfun(@isempty, {el1 el2 el3 el4 al1}))
        error('Error: A boundary selection is empty.');
    end

    %% Set material properties
    mat1 = model.material.create('mat1');
    mat1.propertyGroup('def').set('youngsmodulus', {num2str(ymod)});
    mat1.propertyGroup('def').set('poissonsratio', {num2str(pratio)});
    mat1.propertyGroup('def').set('density', {num2str(rho)});
    mat1.selection.set([elastic]);

    %% Set boundary conditions
    fixed = [el1 el2 el3 el4];
    load = [al1];

    model.physics('solid').feature.create('fix1', 'Fixed', 2);
    model.physics('solid').feature('fix1').selection.set(fixed);

    model.physics('solid').feature.create('bndl1', 'BoundaryLoad', 2);
    model.physics('solid').feature('bndl1').selection.set(load);
    model.physics('solid').feature('bndl1').set('FperArea', {'0' '0' '1'});

    %% Mesh geometry
    mesh1 = model.mesh.create('mesh1', 'geom1');

    ftet1 = mesh1.feature.create('ftet1', 'FreeTet');
    ftet1.selection.set([elastic]);
    ftet1.feature.create('size1', 'Size');
    ftet1.feature('size1').set('hauto', num2str(fine));

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
    pause;

    %% set the loop to iterate over all BEM verts
    nverts = size(verts, 2);
    x = zeros(nverts, nverts);

    for i = 1:nverts
            
        disp([num2str(i) '/' num2str(nverts)])

        locx = verts(1, i);
        locy = verts(2, i);
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
        mem_disp = mphinterp(model, 'w', 'coord', verts);
        x(i,:) = mem_disp;

    end
end