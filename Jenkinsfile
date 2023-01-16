pipeline {
    agent { docker { image 'python:3.10.7-alpine' } }
    stages {
        stage('TEST') {
            steps {
                sh 'python --version'

                timeout(time: 2, unit: 'MINUTES') {
                    sh '''
                        sudo apt-get update -y
                        sudo apt-get install libbluetooth-dev -y
                        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
                        bash miniconda.sh -b -p $HOME/miniconda
                        export PATH="$HOME/miniconda/bin:$PATH"
                        conda init bash
                        conda config --add channels anaconda
                        conda config --add channels conda-forge
                        conda install -y -c conda-forge liblsl
                        pip install -e .[test]
                        python -m pytest --import-mode=append
                    '''
                }
            }
        }
    }
}