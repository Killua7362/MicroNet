pipeline {
    agent any
    stages {
        stage('Pull Git') {
            steps {
                echo 'Building..'
                deleteDir() // Clean the workspace
                checkout scm
            }
        }
        stage('Testing the repo'){
            steps{
                dir("MicroNet"){
                    sh 'ls'
                }
            }
        }
    }
}