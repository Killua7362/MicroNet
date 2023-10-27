pipeline {
    agent any
    stages {
        stage('Pull Git') {
            steps {
                echo 'Building..'
                steps{
                    checkout scm
                }
            }
        }
        stage('Testing the repo'){
            steps{
                dir("MicroNet"){
                    sh 'cat server.py'
                }
            }
        }
    }
}