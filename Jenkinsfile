pipeline {
    agent any
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