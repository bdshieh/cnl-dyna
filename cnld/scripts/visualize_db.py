



if __name__ == '__main__':

    import papermill
    import sys
    import subprocess

    papermill.execute_notebook(
    '../notebooks/visualization/visualize_db.ipynb',
    'temp.ipynb',
    parameters=dict(file=sys.argv[1])
    )

    subprocess.call('jupyter lab temp.ipynb')

