pipeline {
    agent {
        // Equivalent to "docker build -f Dockerfile.build --build-arg version=1.0.2 ./build/
        dockerfile {
            filename 'DockerfileLinux'
            dir 'dockerfiles'
            label 'test-linux'
        }
    }
    stages {
        stage('TEST') {
            steps {
                sh 'python3 --version'

                timeout(time: 10, unit: 'MINUTES') {
                    sh '''
                        pip3 install -e .[test]
                        python3 -m pytest --import-mode=append
                    '''
                }
            }
        }
    }
}