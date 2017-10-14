function isdefault= mv_set_default(cfg,tag,value)
% If the field cfg.tag does not exist, it is set to cfg.tag = value. If it
% already exists, it is left unchanged.

  stridx= strfind(tag,'.'); % check whether a substruct is referenced
  if ~isempty(stridx)
      tagstart= ['.' tag(1:stridx(end)-1)];
      tagend= tag(stridx(end)+1:end);
  else
      tagstart= [];
      tagend= tag;
  end
  if ~evalin('caller',['isfield('  inputname(1) tagstart ',''' tagend ''')'])
    eval(['cfg.' tag ' = value;'])
    % Transfer new value to calling function
    assignin('caller',inputname(1),cfg);    
    isdefault= 1;
  else
    isdefault= 0;
  end
end

