pipeline {
    agent {
        // tests on ubuntu:latest docker image with necessary python dependencies
        dockerfile {
            filename 'DockerfileLinux'
            dir 'dockerfiles'
        }
    }
    stages {
        stage('TEST') {
            steps {
                sh 'python3 --version'

                timeout(time: 10, unit: 'MINUTES') {
                    sh '''
                        pip3 install -e .[test]
                        python3 -m pytest --import-mode=append --junitxml=test-results.xml
                    '''
                }
            }
        }
    }
    post {
        always {
            junit 'test-results.xml'
        }
    }
}