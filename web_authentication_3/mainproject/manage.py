#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'my_project.settings')
    #DJANGO_SETTINGS_MODULE is a env variable it sets to settings module in my_project directory it  goes to my_project.settings file
    try:
       
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        #when django installed automatically instally all its packages if not it shows error
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    #execute_from_command_line is a method
    execute_from_command_line(sys.argv)
    #sys.argv is  list contains command line arguments when it is passes in a method called execute_from_command_line then it will execute 
    #runserver command
    #runserver automatically binds to local host (this machine)127.0.0.1 port number 8000
    #localhost is special to send the network packets to same device



if __name__ == '__main__':
    main()
#when manage.py run as main program then only this main() function will executes
